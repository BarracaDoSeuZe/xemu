/*
 * Experimental persistent caches for the xemu NV2A Vulkan backend.
 * LGPL-2.0-or-later; based on xemu's Vulkan source conventions.
 * This file is used only by glsl.c and draw.c (static inline helpers).
 * Cache data is untrusted and never required for correct emulation.
 */
#ifndef XEMU_VK_DISK_CACHE_V1_H
#define XEMU_VK_DISK_CACHE_V1_H

#include <glib.h>
#include <glib/gstdio.h>
#include "ui/xemu-settings.h"
#include <vulkan/vulkan.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

#define XEMU_VK_SPV_MAX_BYTES (8u * 1024u * 1024u)
#define XEMU_VK_PIPELINE_MAX_BYTES (64u * 1024u * 1024u)
#define XEMU_VK_SPV_HEADER_BYTES 80u

static inline gboolean xemu_vk_disk_cache_enabled(void)
{
    const char *env = g_getenv("XEMU_VK_DISK_CACHE_DISABLE");
    /* Share OpenGL's "Cache shaders to disk" setting. Keep the env override
     * for debugging or clearing the cache without changing user settings. */
    return g_config.perf.cache_shaders &&
           !(env && strcmp(env, "1") == 0);
}

static inline char *xemu_vk_disk_cache_path(const char *subdir,
                                            const char *name)
{
    if (!xemu_vk_disk_cache_enabled()) {
        return NULL;
    }
    /* Follow the OpenGL backend: store shaders next to xemu.toml under
     * xemu_settings_get_base_path()/shaders. Vulkan has its own versioned
     * subtree so neither backend can mistake the other's files for its own. */
    const char *base_dir = xemu_settings_get_base_path();
    if (!base_dir || !*base_dir) {
        return NULL;
    }
    char *dir = g_build_filename(base_dir, "shaders", "vulkan", "v1",
                                 subdir, NULL);
    if (!dir) {
        return NULL;
    }
    if (g_mkdir_with_parents(dir, 0700) != 0) {
        g_free(dir);
        return NULL;
    }
    char *path = g_build_filename(dir, name, NULL);
    g_free(dir);
    return path;
}

/* Hash the complete GLSL plus all compiler options which affect SPIR-V. */
static inline char *xemu_vk_spv_cache_path(const VkPhysicalDeviceProperties *p,
                                           unsigned stage, unsigned client,
                                           unsigned target, gboolean debug,
                                           const char *glsl)
{
    if (!xemu_vk_disk_cache_enabled()) {
        return NULL;
    }
    GChecksum *sum = g_checksum_new(G_CHECKSUM_SHA256);
    if (!sum) {
        return NULL;
    }
    char *options = g_strdup_printf(
        "xemu-nv2a-spv-cache-v1|stage=%u|client=%u|target=%u|debug=%u|"
        "vendor=%08x|device=%08x|driver=%08x|",
        stage, client, target, debug ? 1u : 0u,
        p->vendorID, p->deviceID, p->driverVersion);
    g_checksum_update(sum, (const guchar *)options, strlen(options));
    g_checksum_update(sum, (const guchar *)glsl, strlen(glsl));
    /* Just like the OpenGL shader cache, shard files by a hash prefix
     * rather than placing thousands of shaders in a single directory. */
    const char *hash = g_checksum_get_string(sum);
    char prefix[5];
    memcpy(prefix, hash, 4);
    prefix[4] = '\0';
    char *subdir = g_build_filename("spirv", prefix, NULL);
    char *name = g_strdup_printf("%s.spvbin", hash);
    char *path = xemu_vk_disk_cache_path(subdir, name);
    g_free(subdir);
    g_free(name);
    g_free(options);
    g_checksum_free(sum);
    return path;
}

static inline gboolean xemu_vk_spv_valid(const guint8 *data, gsize size)
{
    guint32 magic;
    if (!data || size < 20 || size > XEMU_VK_SPV_MAX_BYTES || (size & 3u)) {
        return FALSE;
    }
    memcpy(&magic, data, sizeof(magic));
    return magic == 0x07230203u;
}

/* On disk: 8-byte magic + 4-byte size + 4 reserved + 64 ASCII hex digest
 * followed by SPIR-V bytes. The payload checksum catches partial writes. */
static inline GByteArray *xemu_vk_spv_cache_load(const char *path)
{
    if (!path) {
        return NULL;
    }
    GStatBuf statbuf;
    if (g_stat(path, &statbuf) != 0 || statbuf.st_size < 0 ||
        (guint64)statbuf.st_size >
            (guint64)XEMU_VK_SPV_HEADER_BYTES + XEMU_VK_SPV_MAX_BYTES) {
        return NULL;
    }
    gchar *bytes = NULL;
    gsize size = 0;
    if (!g_file_get_contents(path, &bytes, &size, NULL)) {
        return NULL;
    }
    GByteArray *result = NULL;
    if (size >= XEMU_VK_SPV_HEADER_BYTES &&
        memcmp(bytes, "XEVKSPV1", 8) == 0) {
        guint32 le_length;
        memcpy(&le_length, bytes + 8, sizeof(le_length));
        guint32 payload_size = GUINT32_FROM_LE(le_length);
        const guint8 *payload = (const guint8 *)bytes + XEMU_VK_SPV_HEADER_BYTES;
        if (payload_size == size - XEMU_VK_SPV_HEADER_BYTES &&
            xemu_vk_spv_valid(payload, payload_size)) {
            gchar *digest = g_compute_checksum_for_data(
                G_CHECKSUM_SHA256, payload, payload_size);
            if (digest && memcmp(bytes + 16, digest, 64) == 0) {
                result = g_byte_array_sized_new(payload_size);
                g_byte_array_append(result, payload, payload_size);
            }
            g_free(digest);
        }
    }
    g_free(bytes);
    return result;
}

static inline gboolean xemu_vk_spv_cache_save(const char *path,
                                               const GByteArray *spv)
{
    if (!path || !spv || !xemu_vk_spv_valid(spv->data, spv->len)) {
        return FALSE;
    }
    guint8 header[XEMU_VK_SPV_HEADER_BYTES] = { 0 };
    memcpy(header, "XEVKSPV1", 8);
    guint32 le_length = GUINT32_TO_LE(spv->len);
    memcpy(header + 8, &le_length, sizeof(le_length));
    char *digest = g_compute_checksum_for_data(G_CHECKSUM_SHA256,
                                               spv->data, spv->len);
    if (!digest) {
        return FALSE;
    }
    memcpy(header + 16, digest, 64);
    g_free(digest);
    GByteArray *file = g_byte_array_sized_new(XEMU_VK_SPV_HEADER_BYTES + spv->len);
    g_byte_array_append(file, header, sizeof(header));
    g_byte_array_append(file, spv->data, spv->len);
    gboolean ok = g_file_set_contents(path, (const char *)file->data,
                                       file->len, NULL);
    g_byte_array_unref(file);
    return ok;
}

static inline char *xemu_vk_pipeline_cache_path(
    const VkPhysicalDeviceProperties *p)
{
    char uuid[VK_UUID_SIZE * 2 + 1];
    for (size_t i = 0; i < VK_UUID_SIZE; ++i) {
        g_snprintf(uuid + i * 2, 3, "%02x", (unsigned)p->pipelineCacheUUID[i]);
    }
    uuid[sizeof(uuid) - 1] = '\0';
    char *name = g_strdup_printf("%08x-%08x-%08x-%s.bin",
                                  p->vendorID, p->deviceID, p->driverVersion, uuid);
    char *path = xemu_vk_disk_cache_path("pipelines", name);
    g_free(name);
    return path;
}

/* Vulkan-defined pipeline cache header, version 1, minimum 32 bytes.
 * Reject mismatched driver/device caches before giving them to the ICD. */
static inline gboolean xemu_vk_pipeline_cache_blob_valid(
    const guint8 *bytes, gsize size, const VkPhysicalDeviceProperties *p)
{
    if (!bytes || size < 32 || size > XEMU_VK_PIPELINE_MAX_BYTES) {
        return FALSE;
    }
    guint32 header[4];
    memcpy(header, bytes, sizeof(header));
    return header[0] >= 32 && header[0] <= size &&
           header[1] == VK_PIPELINE_CACHE_HEADER_VERSION_ONE &&
           header[2] == p->vendorID && header[3] == p->deviceID &&
           memcmp(bytes + 16, p->pipelineCacheUUID, VK_UUID_SIZE) == 0;
}

static inline GByteArray *xemu_vk_pipeline_cache_load(
    const VkPhysicalDeviceProperties *p)
{
    char *path = xemu_vk_pipeline_cache_path(p);
    if (!path) {
        return NULL;
    }
    GStatBuf statbuf;
    if (g_stat(path, &statbuf) != 0 || statbuf.st_size < 0 ||
        (guint64)statbuf.st_size > XEMU_VK_PIPELINE_MAX_BYTES) {
        g_free(path);
        return NULL;
    }
    gchar *bytes = NULL;
    gsize size = 0;
    GByteArray *result = NULL;
    if (g_file_get_contents(path, &bytes, &size, NULL) &&
        xemu_vk_pipeline_cache_blob_valid((const guint8 *)bytes, size, p)) {
        result = g_byte_array_new_take((guint8 *)bytes, size);
        bytes = NULL;
    }
    g_free(bytes);
    g_free(path);
    return result;
}

static inline void xemu_vk_pipeline_cache_save(
    VkDevice device, VkPipelineCache cache, const VkPhysicalDeviceProperties *p)
{
    if (!xemu_vk_disk_cache_enabled() || cache == VK_NULL_HANDLE) {
        return;
    }
    size_t size = 0;
    if (vkGetPipelineCacheData(device, cache, &size, NULL) != VK_SUCCESS ||
        size < 32 || size > XEMU_VK_PIPELINE_MAX_BYTES) {
        return;
    }
    guint8 *bytes = g_malloc(size);
    if (vkGetPipelineCacheData(device, cache, &size, bytes) == VK_SUCCESS &&
        xemu_vk_pipeline_cache_blob_valid(bytes, size, p)) {
        char *path = xemu_vk_pipeline_cache_path(p);
        if (path && g_file_set_contents(path, (const char *)bytes, size, NULL)) {
            fprintf(stderr, "xemu/vk: saved pipeline cache (%zu bytes)\n", size);
        }
        g_free(path);
    }
    g_free(bytes);
}

#endif /* XEMU_VK_DISK_CACHE_V1_H */
