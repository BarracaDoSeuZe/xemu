/*
 * Geforce NV2A PGRAPH Vulkan Renderer
 *
 * Copyright (c) 2024 Matt Borgerson
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, see <http://www.gnu.org/licenses/>.
 */
#include "renderer.h"

/* Private low-memory variant based on upstream PRs #2996 and #3004.
 * Keep the Xbox's MAX_BATCH_LENGTH unchanged.  Only the transient host
 * buffers and their device-side counterparts are sized more conservatively.
 */
#define XEMU_VK_MIB ((VkDeviceSize)1024 * 1024)


/* Texture uploads share the graphics command stream instead of requiring one
 * GPU submission and host wait per texture.  A persistent, mapped slab arena
 * gives each pending copy a non-overlapping source region.  Reuse is legal
 * only after pgraph_vk_finish() has waited on the batch completion fence.
 *
 * These buffers are host resources, not the emulated Xbox VRAM. */
struct TextureUploadChunk {
    struct TextureUploadChunk *next;
    VkBuffer buffer;
    VmaAllocation allocation;
    uint8_t *mapped;
    VkDeviceSize size;
    VkDeviceSize used;
};

/* 1 MiB chunks avoid a large upfront reservation on an iGPU. */
#define XEMU_VK_UPLOAD_SLAB (1 * XEMU_VK_MIB)
#define XEMU_VK_UPLOAD_RETAIN_LIMIT (32 * XEMU_VK_MIB)

static VkDeviceSize texture_upload_alignment(PGRAPHVkState *r)
{
    /* A VkBufferImageCopy offset must satisfy compressed block alignment
     * (up to 16 bytes), and flushes must not share non-coherent atoms. */
    return MAX((VkDeviceSize)16,
               r->device_props.limits.nonCoherentAtomSize);
}

static TextureUploadChunk *create_texture_upload_chunk(PGRAPHVkState *r,
                                                        VkDeviceSize minimum)
{
    VkDeviceSize slab = XEMU_VK_UPLOAD_SLAB;
    VkDeviceSize align = texture_upload_alignment(r);
    VkDeviceSize size = ROUND_UP(MAX(slab, minimum), align);
    TextureUploadChunk *chunk = g_new0(TextureUploadChunk, 1);
    VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VmaAllocationCreateInfo alloc_info = {
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                 VMA_ALLOCATION_CREATE_MAPPED_BIT,
    };
    VmaAllocationInfo info = { 0 };
    VkResult result = vmaCreateBuffer(r->allocator, &buffer_info, &alloc_info,
                                      &chunk->buffer, &chunk->allocation, &info);
    VK_CHECK(result);
    assert(info.pMappedData != NULL);
    chunk->mapped = info.pMappedData;
    chunk->size = size;
    if (r->texture_upload_tail) {
        r->texture_upload_tail->next = chunk;
    } else {
        r->texture_upload_chunks = chunk;
    }
    r->texture_upload_tail = chunk;
    return chunk;
}

TextureUploadSlice pgraph_vk_reserve_texture_upload(PGRAPHState *pg,
                                                     VkDeviceSize bytes)
{
    PGRAPHVkState *r = pg->vk_renderer_state;
    assert(bytes > 0);
    VkDeviceSize alignment = texture_upload_alignment(r);
    TextureUploadChunk *chunk = r->texture_upload_chunks;
    VkDeviceSize pos = 0;
    while (chunk) {
        pos = ROUND_UP(chunk->used, alignment);
        if (pos <= chunk->size && bytes <= chunk->size - pos) {
            break;
        }
        chunk = chunk->next;
    }
    if (!chunk) {
        chunk = create_texture_upload_chunk(r, bytes);
        pos = 0;
    }
    chunk->used = pos + bytes;
    assert(pos % alignment == 0);
    assert(chunk->used <= chunk->size);
    r->texture_uploads_in_batch++;
    r->texture_upload_bytes_in_batch += bytes;
    return (TextureUploadSlice){
        .buffer = chunk->buffer,
        .allocation = chunk->allocation,
        .mapped = chunk->mapped + pos,
        .offset = pos,
    };
}

void pgraph_vk_flush_texture_upload(PGRAPHState *pg,
                                    TextureUploadSlice slice,
                                    VkDeviceSize bytes)
{
    PGRAPHVkState *r = pg->vk_renderer_state;
    VK_CHECK(vmaFlushAllocation(r->allocator, slice.allocation, slice.offset,
                                bytes));
}

/* This function must only run AFTER the graphics submission fence signals.
 * Keep at most 32 MiB for reuse on iGPUs; release the excess only when
 * nothing can still access it. */
void pgraph_vk_recycle_texture_uploads(PGRAPHState *pg)
{
    PGRAPHVkState *r = pg->vk_renderer_state;
    /* Opt-in diagnostics.  Never emit per-frame logs on the normal path. */
    static int report_uploads = -1;
    if (report_uploads < 0) {
        const char *env = getenv("XEMU_VK_UPLOAD_STATS");
        report_uploads = env && strcmp(env, "1") == 0;
    }
    if (report_uploads && r->texture_uploads_in_batch) {
        fprintf(stderr,
                "xemu/vk: batched %u texture uploads (%llu KiB), submit %u\n",
                r->texture_uploads_in_batch,
                (unsigned long long)(r->texture_upload_bytes_in_batch / 1024),
                r->submit_count);
    }
    TextureUploadChunk **link = &r->texture_upload_chunks;
    TextureUploadChunk *last = NULL;
    VkDeviceSize retained = 0;
    while (*link) {
        TextureUploadChunk *chunk = *link;
        if (retained && chunk->size > XEMU_VK_UPLOAD_RETAIN_LIMIT -
                                        MIN(retained, XEMU_VK_UPLOAD_RETAIN_LIMIT)) {
            *link = chunk->next;
            vmaDestroyBuffer(r->allocator, chunk->buffer, chunk->allocation);
            g_free(chunk);
            continue;
        }
        chunk->used = 0;
        retained += chunk->size;
        last = chunk;
        link = &chunk->next;
    }
    r->texture_upload_tail = last;
    r->texture_uploads_in_batch = 0;
    r->texture_upload_bytes_in_batch = 0;
}

static void destroy_texture_upload_arena(PGRAPHVkState *r)
{
    TextureUploadChunk *chunk = r->texture_upload_chunks;
    while (chunk) {
        TextureUploadChunk *next = chunk->next;
        vmaDestroyBuffer(r->allocator, chunk->buffer, chunk->allocation);
        g_free(chunk);
        chunk = next;
    }
    r->texture_upload_chunks = r->texture_upload_tail = NULL;
}
#define XEMU_VK_MAX_TRANSIENT_BUFFER (256 * XEMU_VK_MIB)

/* V7: Keep the guest-visible NV2A limits unchanged. Only the host Vulkan
 * vertex/index/uniform streaming pairs may grow. Compute and surface staging
 * keep their V5/V6 sizes until their independent call sites can be audited. */
static VkDeviceSize transient_budget_available(PGRAPHVkState *r)
{
    VkPhysicalDeviceMemoryProperties props;
    VmaBudget budgets[VK_MAX_MEMORY_HEAPS] = { 0 };
    VkDeviceSize host_free = 0, device_free = 0;

    vkGetPhysicalDeviceMemoryProperties(r->physical_device, &props);
    vmaGetHeapBudgets(r->allocator, budgets);

    for (uint32_t i = 0; i < props.memoryTypeCount; i++) {
        uint32_t heap = props.memoryTypes[i].heapIndex;
        VkMemoryPropertyFlags flags = props.memoryTypes[i].propertyFlags;
        if (heap >= props.memoryHeapCount) {
            continue;
        }
        VkDeviceSize available = budgets[heap].budget > budgets[heap].usage ?
            budgets[heap].budget - budgets[heap].usage : 0;
        if (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
            host_free = MAX(host_free, available);
        }
        if (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
            device_free = MAX(device_free, available);
        }
    }

    /* The two allocations can share the same underlying heap on an iGPU.
     * This is a heuristic only: VMA allocations remain the authority. */
    if (host_free && device_free) {
        return MIN(host_free, device_free);
    }
    return MAX(host_free, device_free);
}

static VkDeviceSize initial_streaming_capacity(PGRAPHVkState *r,
                                                VkDeviceSize desired,
                                                VkDeviceSize minimum,
                                                VkDeviceSize maximum,
                                                uint32_t budget_divisor)
{
    VkDeviceSize available = transient_budget_available(r);
    VkDeviceSize target = available ? available / budget_divisor : maximum;
    target = MAX(minimum, MIN(target, maximum));
    return MIN(desired, target);
}

/* The upper bounds are based on the original full-batch allocations, not
 * on a fixed 256 MiB limit. Growth is demand-driven and can exceed that
 * size on a GPU with enough memory. */
static VkDeviceSize streaming_maximum(PGRAPHState *pg, int index)
{
    switch (index) {
    case BUFFER_VERTEX_INLINE_STAGING:
        return (VkDeviceSize)NV2A_VERTEXSHADER_ATTRIBUTES *
               NV2A_MAX_BATCH_LENGTH * 4 * sizeof(float) * 10;
    case BUFFER_INDEX_STAGING:
        return (VkDeviceSize)sizeof(pg->inline_elements) * 100;
    case BUFFER_UNIFORM_STAGING:
        return 64 * XEMU_VK_MIB;
    default:
        return 0;
    }
}

static int streaming_device_pair(int index)
{
    switch (index) {
    case BUFFER_VERTEX_INLINE_STAGING: return BUFFER_VERTEX_INLINE;
    case BUFFER_INDEX_STAGING: return BUFFER_INDEX;
    case BUFFER_UNIFORM_STAGING: return BUFFER_UNIFORM;
    default: return -1;
    }
}

static VkDeviceSize clamp_buffer_size(PGRAPHVkState *r, VkDeviceSize desired,
                                      VkDeviceSize cap, bool storage)
{
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(r->physical_device, &mem_props);

    VkDeviceSize largest_heap = 0;
    for (uint32_t i = 0; i < mem_props.memoryHeapCount; i++) {
        if (mem_props.memoryHeaps[i].size > largest_heap) {
            largest_heap = mem_props.memoryHeaps[i].size;
        }
    }

    VkDeviceSize limit = MIN(desired, cap);
    if (largest_heap) {
        limit = MIN(limit, largest_heap / 16);
    }
    if (storage) {
        /* Never raise this cap to a nominal 8 MiB floor: Vulkan's actual
         * maxStorageBufferRange is a hard validity requirement. */
        limit = MIN(limit, r->device_props.limits.maxStorageBufferRange);
    }
    return MAX(limit, 1);
}

static VkResult try_create_buffer(PGRAPHState *pg, StorageBuffer *buffer,
                                  VkDeviceSize size)
{
    PGRAPHVkState *r = pg->vk_renderer_state;
    VkBufferCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = buffer->usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    return vmaCreateBuffer(r->allocator, &info, &buffer->alloc_info,
                           &buffer->buffer, &buffer->allocation, NULL);
}

static void destroy_buffer(PGRAPHState *pg, StorageBuffer *buffer)
{
    PGRAPHVkState *r = pg->vk_renderer_state;

    vmaDestroyBuffer(r->allocator, buffer->buffer, buffer->allocation);
    buffer->buffer = VK_NULL_HANDLE;
    buffer->allocation = VK_NULL_HANDLE;
}

static bool memory_allocation_failed(VkResult result)
{
    return result == VK_ERROR_OUT_OF_DEVICE_MEMORY ||
           result == VK_ERROR_OUT_OF_HOST_MEMORY;
}

/* For buffers copied as a whole (and compute source/destination), retry the
 * *pair* at the same smaller size.  Retrying them independently could make a
 * later vkCmdCopyBuffer copy beyond the destination buffer's capacity. */
static void create_buffer_pair(PGRAPHState *pg, int first, int second)
{
    PGRAPHVkState *r = pg->vk_renderer_state;
    StorageBuffer *a = &r->storage_buffers[first];
    StorageBuffer *b = &r->storage_buffers[second];
    VkDeviceSize requested = MIN(a->buffer_size, b->buffer_size);
    VkDeviceSize size = requested;
    const VkDeviceSize minimum = MIN(size, 8 * XEMU_VK_MIB);

    for (;;) {
        VkResult result = try_create_buffer(pg, a, size);
        if (result == VK_SUCCESS) {
            result = try_create_buffer(pg, b, size);
            if (result == VK_SUCCESS) {
                a->buffer_size = size;
                b->buffer_size = size;
                if (size != requested) {
                    fprintf(stderr,
                            "xemu Vulkan: buffer pair %d/%d allocated %llu MiB "
                            "instead of %llu MiB\n", first, second,
                            (unsigned long long)(size / XEMU_VK_MIB),
                            (unsigned long long)(requested / XEMU_VK_MIB));
                }
                return;
            }
            destroy_buffer(pg, a);
        }
        if (!memory_allocation_failed(result) || size <= minimum) {
            fprintf(stderr,
                    "xemu Vulkan: buffer pair %d/%d failed at %llu bytes "
                    "(vk_result=%d)\n", first, second,
                    (unsigned long long)size, (int)result);
            VK_CHECK(result);
            return;
        }

        VkDeviceSize next = MAX(minimum, size / 2);
        fprintf(stderr,
                "xemu Vulkan: buffer pair %d/%d out of memory at %llu MiB, "
                "retrying at %llu MiB\n", first, second,
                (unsigned long long)(size / XEMU_VK_MIB),
                (unsigned long long)(next / XEMU_VK_MIB));
        size = next;
    }
}

/* BUFFER_VERTEX_RAM maps the complete guest VRAM address space.  Reducing
 * its size is unsafe, because vertex RAM accesses use absolute offsets. */
static void create_required_buffer(PGRAPHState *pg, int index)
{
    PGRAPHVkState *r = pg->vk_renderer_state;
    StorageBuffer *buffer = &r->storage_buffers[index];
    VkResult result = try_create_buffer(pg, buffer, buffer->buffer_size);
    if (result != VK_SUCCESS) {
        fprintf(stderr,
                "xemu Vulkan: required buffer %d failed at %llu bytes "
                "(vk_result=%d)\n", index,
                (unsigned long long)buffer->buffer_size, (int)result);
    }
    VK_CHECK(result);
}

void pgraph_vk_init_buffers(NV2AState *d)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHVkState *r = pg->vk_renderer_state;

    // FIXME: Profile buffer sizes
    VmaAllocationCreateInfo host_alloc_create_info = {
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                 VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
    };
    VmaAllocationCreateInfo device_alloc_create_info = {
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT
    };
    r->storage_buffers[BUFFER_STAGING_DST] = (StorageBuffer){
        .alloc_info = host_alloc_create_info,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .buffer_size = clamp_buffer_size(r, 4096 * 4096 * 4,
                                          64 * XEMU_VK_MIB, false),
    };

    r->storage_buffers[BUFFER_STAGING_SRC] = (StorageBuffer){
        .alloc_info = host_alloc_create_info,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .buffer_size = r->storage_buffers[BUFFER_STAGING_DST].buffer_size,
    };
    r->storage_buffers[BUFFER_COMPUTE_DST] = (StorageBuffer){
        .alloc_info = device_alloc_create_info,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .buffer_size = clamp_buffer_size(r,
                     (VkDeviceSize)(1024 * 10) * (1024 * 10) * 8,
                     XEMU_VK_MAX_TRANSIENT_BUFFER, true),
    };
    r->storage_buffers[BUFFER_COMPUTE_SRC] = (StorageBuffer){
        .alloc_info = device_alloc_create_info,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .buffer_size = r->storage_buffers[BUFFER_COMPUTE_DST].buffer_size,
    };
    r->storage_buffers[BUFFER_INDEX] = (StorageBuffer){
        .alloc_info = device_alloc_create_info,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        .buffer_size = initial_streaming_capacity(
            r, (VkDeviceSize)sizeof(pg->inline_elements) * 100,
            4 * XEMU_VK_MIB, 16 * XEMU_VK_MIB, 256),
    };

    r->storage_buffers[BUFFER_INDEX_STAGING] = (StorageBuffer){
        .alloc_info = host_alloc_create_info,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .buffer_size = r->storage_buffers[BUFFER_INDEX].buffer_size,
    };
    // FIXME: Don't assume that we can render with host mapped buffer
    r->storage_buffers[BUFFER_VERTEX_RAM] = (StorageBuffer){
        .alloc_info = host_alloc_create_info,
        .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        .buffer_size = memory_region_size(d->vram),
    };

    r->bitmap_size = memory_region_size(d->vram) / 4096;
    r->uploaded_bitmap = bitmap_new(r->bitmap_size);
    bitmap_clear(r->uploaded_bitmap, 0, r->bitmap_size);
    r->storage_buffers[BUFFER_VERTEX_INLINE] = (StorageBuffer){
        .alloc_info = device_alloc_create_info,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        .buffer_size = initial_streaming_capacity(
            r, (VkDeviceSize)NV2A_VERTEXSHADER_ATTRIBUTES *
                   NV2A_MAX_BATCH_LENGTH * 4 * sizeof(float) * 10,
            16 * XEMU_VK_MIB, 64 * XEMU_VK_MIB, 64),
    };
    r->storage_buffers[BUFFER_VERTEX_INLINE_STAGING] = (StorageBuffer){
        .alloc_info = host_alloc_create_info,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .buffer_size = r->storage_buffers[BUFFER_VERTEX_INLINE].buffer_size,
    };
    r->storage_buffers[BUFFER_UNIFORM] = (StorageBuffer){
        .alloc_info = device_alloc_create_info,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        .buffer_size = 8 * 1024 * 1024,
    };

    r->storage_buffers[BUFFER_UNIFORM_STAGING] = (StorageBuffer){
        .alloc_info = host_alloc_create_info,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .buffer_size = r->storage_buffers[BUFFER_UNIFORM].buffer_size,
    };
    /* Pair allocations are atomic: if either half fails, discard the
     * successful half and retry both at the same reduced capacity. */
    /* Allocate fixed and modest-size resources before the large transient
     * pairs, so an early oversized allocation does not starve them. */
    create_buffer_pair(pg, BUFFER_STAGING_DST, BUFFER_STAGING_SRC);
    create_buffer_pair(pg, BUFFER_INDEX, BUFFER_INDEX_STAGING);
    create_required_buffer(pg, BUFFER_VERTEX_RAM);
    create_buffer_pair(pg, BUFFER_UNIFORM, BUFFER_UNIFORM_STAGING);
    create_buffer_pair(pg, BUFFER_VERTEX_INLINE,
                       BUFFER_VERTEX_INLINE_STAGING);
    create_buffer_pair(pg, BUFFER_COMPUTE_DST, BUFFER_COMPUTE_SRC);

    fprintf(stderr, "xemu/vk: batched texture staging enabled (1 MiB slabs)\n");

    // FIXME: Add fallback path for device using host mapped memory

    int buffers_to_map[] = { BUFFER_VERTEX_RAM,
                             BUFFER_INDEX_STAGING,
                             BUFFER_VERTEX_INLINE_STAGING,
                             BUFFER_UNIFORM_STAGING };
    for (int i = 0; i < ARRAY_SIZE(buffers_to_map); i++) {
        VK_CHECK(vmaMapMemory(
            r->allocator, r->storage_buffers[buffers_to_map[i]].allocation,
            (void **)&r->storage_buffers[buffers_to_map[i]].mapped));
    }
    pgraph_vk_init_batch_stream_buffers(d);
}

static const int batch_stream_indices[] = {
    BUFFER_INDEX, BUFFER_INDEX_STAGING,
    BUFFER_VERTEX_INLINE, BUFFER_VERTEX_INLINE_STAGING,
    BUFFER_UNIFORM, BUFFER_UNIFORM_STAGING,
};

static bool batch_stream_index(int index)
{
    for (unsigned int i = 0; i < ARRAY_SIZE(batch_stream_indices); i++) {
        if (batch_stream_indices[i] == index) {
            return true;
        }
    }
    return false;
}

void pgraph_vk_init_batch_stream_buffers(NV2AState *d)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHVkState *r = pg->vk_renderer_state;
    StorageBuffer baseline[ARRAY_SIZE(batch_stream_indices)];

    for (unsigned int j = 0; j < ARRAY_SIZE(batch_stream_indices); j++) {
        int idx = batch_stream_indices[j];
        baseline[j] = r->storage_buffers[idx];
        r->batches[0].stream_buffers[idx] = baseline[j];
    }
    for (unsigned int slot = 0; slot < XEMU_VK_BATCH_COUNT; slot++) {
        r->batches[slot].vertex_pages = bitmap_new(r->bitmap_size);
        bitmap_clear(r->batches[slot].vertex_pages, 0, r->bitmap_size);
    }

    /* Reserve less memory for the initially idle second frame. The V7
     * allocator grows each pair on demand once that slot is retired. */
    for (unsigned int slot = 1; slot < XEMU_VK_BATCH_COUNT; slot++) {
        for (unsigned int j = 0; j < ARRAY_SIZE(batch_stream_indices); j++) {
            StorageBuffer clone = baseline[j];
            clone.buffer = VK_NULL_HANDLE;
            clone.allocation = VK_NULL_HANDLE;
            clone.mapped = NULL;
            clone.buffer_offset = 0;
            clone.growth_pressure = 0;
            VkDeviceSize cap =
                batch_stream_indices[j] == BUFFER_VERTEX_INLINE ||
                batch_stream_indices[j] == BUFFER_VERTEX_INLINE_STAGING ?
                    16 * XEMU_VK_MIB :
                batch_stream_indices[j] == BUFFER_INDEX ||
                batch_stream_indices[j] == BUFFER_INDEX_STAGING ?
                    4 * XEMU_VK_MIB : 8 * XEMU_VK_MIB;
            clone.buffer_size = MIN(clone.buffer_size, cap);
            r->storage_buffers[batch_stream_indices[j]] = clone;
        }
        create_buffer_pair(pg, BUFFER_INDEX, BUFFER_INDEX_STAGING);
        create_buffer_pair(pg, BUFFER_VERTEX_INLINE,
                           BUFFER_VERTEX_INLINE_STAGING);
        create_buffer_pair(pg, BUFFER_UNIFORM, BUFFER_UNIFORM_STAGING);
        int host_indices[] = { BUFFER_INDEX_STAGING,
                               BUFFER_VERTEX_INLINE_STAGING,
                               BUFFER_UNIFORM_STAGING };
        for (unsigned int j = 0; j < ARRAY_SIZE(host_indices); j++) {
            StorageBuffer *host = &r->storage_buffers[host_indices[j]];
            VK_CHECK(vmaMapMemory(r->allocator, host->allocation,
                                  (void **)&host->mapped));
        }
        for (unsigned int j = 0; j < ARRAY_SIZE(batch_stream_indices); j++) {
            int idx = batch_stream_indices[j];
            r->batches[slot].stream_buffers[idx] = r->storage_buffers[idx];
            r->storage_buffers[idx] = baseline[j];
        }
    }
    fprintf(stderr, "xemu/vk: 2 Vulkan graphics batches available "
                    "(separate streaming resources)\n");
}

void pgraph_vk_finalize_buffers(NV2AState *d)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHVkState *r = pg->vk_renderer_state;
    /* The renderer must drain its graphics queue before destroying these
     * persistently mapped upload slabs. */
    assert(!r->in_command_buffer && !r->in_aux_command_buffer);
    pgraph_vk_save_batch_state(r);
    for (unsigned int slot = 0; slot < XEMU_VK_BATCH_COUNT; slot++) {
        VkGraphicsBatchSlot *batch = &r->batches[slot];
        r->texture_upload_chunks = batch->upload_chunks;
        r->texture_upload_tail = batch->upload_tail;
        destroy_texture_upload_arena(r);
        for (unsigned int j = 0; j < ARRAY_SIZE(batch_stream_indices); j++) {
            StorageBuffer *buf =
                &batch->stream_buffers[batch_stream_indices[j]];
            if (buf->mapped) {
                vmaUnmapMemory(r->allocator, buf->allocation);
            }
            destroy_buffer(pg, buf);
        }
        g_free(batch->vertex_pages);
        batch->vertex_pages = NULL;
    }
    for (int i = 0; i < BUFFER_COUNT; i++) {
        if (batch_stream_index(i)) {
            continue;
        }
        if (r->storage_buffers[i].mapped) {
            vmaUnmapMemory(r->allocator, r->storage_buffers[i].allocation);
        }
        destroy_buffer(pg, &r->storage_buffers[i]);
    }
    r->texture_upload_chunks = r->texture_upload_tail = NULL;
    g_free(r->uploaded_bitmap);
    r->uploaded_bitmap = NULL;
}
/* Reserve a larger *pair* only at a drained command-buffer boundary.
 * We keep the old allocations alive until both new ones are created and the
 * replacement staging allocation is mapped. A failure leaves the old pair
 * intact; callers may treat an essential oversized draw as a fatal error.
 */
bool pgraph_vk_reserve_streaming_capacity(PGRAPHState *pg, int staging_index,
                                          VkDeviceSize required)
{
    PGRAPHVkState *r = pg->vk_renderer_state;
    int device_index = streaming_device_pair(staging_index);
    if (device_index < 0) {
        return false;
    }

    StorageBuffer *host = &r->storage_buffers[staging_index];
    StorageBuffer *device = &r->storage_buffers[device_index];
    VkDeviceSize maximum = streaming_maximum(pg, staging_index);
    if (required <= host->buffer_size) {
        return true;
    }
    if (required > maximum || required == 0) {
        return false;
    }

    /* A previous pgraph_vk_finish() must have submitted all draws using
     * the old pair and reset the staging cursor. */
    assert(!r->in_command_buffer && !r->in_aux_command_buffer);
    assert(host->buffer_offset == 0);

    VkDeviceSize next = host->buffer_size <= maximum / 2 ?
                        host->buffer_size * 2 : maximum;
    next = MIN(maximum, MAX(next, required));

    VkDeviceSize available = transient_budget_available(r);
    if (available) {
        /* Treat the budget as guidance, never as a hard capacity guarantee.
         * Leave headroom for images and other resources, especially when
         * host-visible and device-local memory share the same heap. */
        VkDeviceSize comfortable = available / 4;
        next = MAX(required, MIN(next, comfortable));
    }

    while (next >= required) {
        StorageBuffer replacement_host = *host;
        StorageBuffer replacement_device = *device;
        replacement_host.buffer = VK_NULL_HANDLE;
        replacement_host.allocation = VK_NULL_HANDLE;
        replacement_host.mapped = NULL;
        replacement_host.buffer_offset = 0;
        replacement_device.buffer = VK_NULL_HANDLE;
        replacement_device.allocation = VK_NULL_HANDLE;
        replacement_device.mapped = NULL;
        replacement_device.buffer_offset = 0;

        VkResult result = try_create_buffer(pg, &replacement_device, next);
        if (result == VK_SUCCESS) {
            result = try_create_buffer(pg, &replacement_host, next);
        }
        if (result == VK_SUCCESS) {
            result = vmaMapMemory(r->allocator, replacement_host.allocation,
                                  (void **)&replacement_host.mapped);
        }

        if (result == VK_SUCCESS) {
            /* A rare full-device wait is safer than destroying a buffer
             * referenced by an older transfer or descriptor submission. */
            VK_CHECK(vkDeviceWaitIdle(r->device));
            if (host->mapped) {
                vmaUnmapMemory(r->allocator, host->allocation);
            }
            destroy_buffer(pg, host);
            destroy_buffer(pg, device);
            replacement_host.buffer_size = next;
            replacement_device.buffer_size = next;
            replacement_host.growth_pressure = 0;
            *host = replacement_host;
            *device = replacement_device;
            fprintf(stderr,
                    "xemu Vulkan V7: grew streaming pair %d/%d to %llu MiB\n",
                    staging_index, device_index,
                    (unsigned long long)(next / XEMU_VK_MIB));
            return true;
        }

        if (replacement_host.mapped) {
            vmaUnmapMemory(r->allocator, replacement_host.allocation);
        }
        if (replacement_host.buffer) {
            destroy_buffer(pg, &replacement_host);
        }
        if (replacement_device.buffer) {
            destroy_buffer(pg, &replacement_device);
        }
        if (!memory_allocation_failed(result) || next == required) {
            fprintf(stderr,
                    "xemu Vulkan V7: cannot grow pair %d/%d to %llu bytes "
                    "(VkResult=%d)\n",
                    staging_index, device_index,
                    (unsigned long long)required, (int)result);
            return false;
        }
        VkDeviceSize smaller = MAX(required, next / 2);
        if (smaller == next) {
            break;
        }
        next = smaller;
    }
    return false;
}

/* Repeated nearly-full streaming buffers indicate a workload that benefits
 * from more space, even if no single draw exceeds the current allocation.
 * Only the caller of this helper may decide to grow at a safe boundary. */
bool pgraph_vk_streaming_pressure(PGRAPHState *pg, int staging_index,
                                  VkDeviceSize used)
{
    PGRAPHVkState *r = pg->vk_renderer_state;
    if (streaming_device_pair(staging_index) < 0) {
        return false;
    }
    StorageBuffer *buf = &r->storage_buffers[staging_index];
    if (used < buf->buffer_size / 4 * 3) {
        buf->growth_pressure = 0;
        return false;
    }
    if (++buf->growth_pressure < 8) {
        return false;
    }
    buf->growth_pressure = 0;
    return true;
}

bool pgraph_vk_buffer_has_space_for(PGRAPHState *pg, int index,
                                    VkDeviceSize size,
                                    VkDeviceAddress alignment)
{
    PGRAPHVkState *r = pg->vk_renderer_state;
    StorageBuffer *b = &r->storage_buffers[index];
    /* Return false instead of aborting: the caller can drain the pending
     * commands and enlarge an eligible host/device pair before writing. */
    if (size > b->buffer_size) {
        return false;
    }
    VkDeviceSize offset = ROUND_UP(b->buffer_offset, alignment);
    return offset <= b->buffer_size - size;
}
VkDeviceSize pgraph_vk_append_to_buffer(PGRAPHState *pg, int index, void **data,
                                        VkDeviceSize *sizes, size_t count,
                                        VkDeviceAddress alignment)
{
    PGRAPHVkState *r = pg->vk_renderer_state;

    VkDeviceSize total_size = 0;
    for (int i = 0; i < count; i++) {
        total_size += sizes[i];
    }
    assert(pgraph_vk_buffer_has_space_for(pg, index, total_size, alignment));
    StorageBuffer *b = &r->storage_buffers[index];
    VkDeviceSize starting_offset = ROUND_UP(b->buffer_offset, alignment);

    assert(b->mapped);

    for (int i = 0; i < count; i++) {
        b->buffer_offset = ROUND_UP(b->buffer_offset, alignment);
        if (b->buffer_offset > b->buffer_size ||
            sizes[i] > b->buffer_size - b->buffer_offset) {
            fprintf(stderr,
                    "xemu Vulkan: append exceeds buffer %d (offset=%zu, "
                    "length=%llu, capacity=%zu)\n",
                    index, b->buffer_offset,
                    (unsigned long long)sizes[i], b->buffer_size);
            abort();
        }
        memcpy(b->mapped + b->buffer_offset, data[i], sizes[i]);
        b->buffer_offset += sizes[i];
    }

    return starting_offset;
}
