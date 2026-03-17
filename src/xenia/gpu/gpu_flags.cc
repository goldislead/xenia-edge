/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/gpu_flags.h"

#include "xenia/base/logging.h"
#include "xenia/ui/renderdoc_api.h"

DEFINE_bool(use_50Hz_mode, false, "Enables usage of PAL-50 mode.", "Video");

DEFINE_path(trace_gpu_prefix, "scratch/gpu/",
            "Prefix path for GPU trace files.", "GPU");
DEFINE_bool(trace_gpu_stream, false, "Trace all GPU packets.", "GPU");

DEFINE_path(
    dump_shaders, "",
    "For shader debugging, path to dump GPU shaders to as they are compiled.",
    "GPU");

DEFINE_bool(guest_display_refresh_cap, true,
            "Control guest vblank timing.\n"
            "  true: Fixed rate vblanks (50Hz PAL, 60Hz NTSC based on "
            "use_50Hz_mode).\n"
            "  false: Unlimited vblanks, allows the guest to run as fast as "
            "possible.",
            "GPU");

DEFINE_uint64(
    framerate_limit, 0,
    "Host frame rate limit in FPS. 0 = unlimited.\n"
    "Throttles presentation without affecting guest vblank timing.\n"
    "Guest vblanks are controlled by use_50Hz_mode (50Hz PAL, 60Hz NTSC).",
    "GPU");
UPDATE_from_uint64(framerate_limit, 2024, 8, 31, 20, 60);

void SetGuestDisplayRefreshCap(bool value) {
  OVERRIDE_bool(guest_display_refresh_cap, value);
}

void SetFramerateLimit(uint64_t value) {
  OVERRIDE_uint64(framerate_limit, value);
}

DEFINE_bool(
    gpu_allow_invalid_fetch_constants, true,
    "Allow texture and vertex fetch constants with invalid type - generally "
    "unsafe because the constant may contain completely invalid values, but "
    "may be used to bypass fetch constant type errors in certain games until "
    "the real reason why they're invalid is found.",
    "GPU");
DEFINE_bool(
    gpu_allow_invalid_upload_range, false,
    "Allows games to read data from pages that are marked as no access.",
    "GPU");

DEFINE_bool(
    non_seamless_cube_map, true,
    "Disable filtering between cube map faces near edges where possible "
    "(Vulkan with VK_EXT_non_seamless_cube_map) to reproduce the Direct3D 9 "
    "behavior.",
    "GPU");

// Extremely bright screen borders in 4D5307E6.
// Reading between texels with half-pixel offset in 58410954.
DEFINE_bool(
    half_pixel_offset, true,
    "Enable support of vertex half-pixel offset (D3D9 PA_SU_VTX_CNTL "
    "PIX_CENTER). Generally games are aware of the half-pixel offset, and "
    "having this enabled is the correct behavior (disabling this may "
    "significantly break post-processing in some games), but in certain games "
    "it might have been ignored, resulting in slight blurriness of UI "
    "textures, for instance, when they are read between texels rather than "
    "at texel centers, or the leftmost/topmost pixels may not be fully covered "
    "when MSAA is used with fullscreen passes.",
    "GPU");

DEFINE_bool(occlusion_query_enable, false,
            "Use real host occlusion queries instead of faking results.\n"
            "Still experimental. Some games improve a lot. Others may "
            "experience missing objects, slow lens flare updates, or hangs.",
            "GPU");
DEFINE_bool(
    occlusion_query_fast, true,
    "Enable the fast path for hardware occlusion queries.\n"
    "Uses cached delta writes and a one submission delay so guest polling "
    "doesn't stall while the real resolve catches up. Some games need this "
    "disabled to handle occlusion correctly.",
    "GPU");
DEFINE_int32(occlusion_query_fake_lower_threshold, 80,
             "Lower end of the fake sample count value written on "
             "EVENT_WRITE_ZPD when real occlusion queries are off.\n"
             "-1 writes nothing, resulting in some games that sit and hang.\n"
             "0 means the fake result stays fully occluded.",
             "GPU");
DEFINE_int32(occlusion_query_fake_upper_threshold, 100,
             "Upper end of the fake sample count value written on "
             "EVENT_WRITE_ZPD when real occlusion queries are off.\n"
             "Keep this higher than occlusion_query_fake_lower_threshold.\n"
             "Ignored if occlusion_query_fake_lower_threshold is -1.",
             "GPU");
DEFINE_int32(
    occlusion_query_fast_cached_delta, 1,
    "When hardware occlusion is enabled and fast readback is active, use this "
    "delta for speculative writes until a real result is cached.\n"
    "Leave this at 1 unless a game clearly needs a different visible fallback.",
    "GPU");
DEFINE_bool(
    occlusion_query_strict_immediate_sentinel_clear, false,
    "In strict occlusion query mode, write a speculative result to the END\n"
    "record immediately rather than leaving the 0xFFFFFEED pending sentinel\n"
    "until the GPU result arrives. Intended for titles that read report memory\n"
    "directly without going through GetData and stall on the sentinel value.\n"
    "Disables S_FALSE signaling for those queries. The real GPU result still\n"
    "overwrites the speculative value when it arrives.",
    "GPU");
DEFINE_int32(
    occlusion_query_strict_retire_max_stalls, 16,
    "Maximum number of consecutive failed retirement attempts before a stuck\n"
    "strict-mode query handle is abandoned. Prevents indefinite stalling when\n"
    "a GPU query submission never completes. Values less than or equal to 0\n"
    "use the default of 16.",
    "GPU");
DEFINE_bool(occlusion_query_log, false,
            "Log hardware occlusion query lifetime and summary stats.\n"
            "Only useful for debugging and testing.",
            "GPU");
DEFINE_int32(occlusion_query_pool_size, 8192,
             "Number of host occlusion query slots to allocate.\n"
             "The default should be enough for most games. Raise it only if a "
             "game burns through the pool and starts stalling or hanging.",
             "GPU");

void SetOcclusionQueryEnable(bool value) {
  OVERRIDE_bool(occlusion_query_enable, value);
}

uint32_t GetGuestVblankRateHz() { return cvars::use_50Hz_mode ? 50 : 60; }

DEFINE_bool(
    gpu_debug_markers, false,
    "Insert debug markers into GPU command streams for tools like RenderDoc. "
    "Annotates draw calls with Xbox 360 GPU context (primitive type, shader "
    "hashes, vertex count, etc). Automatically enabled when RenderDoc is "
    "detected. Has minimal overhead when disabled.",
    "GPU");

bool IsGpuDebugMarkersEnabled() {
  // Cache the result - RenderDoc detection only needs to happen once.
  static bool cached = false;
  static bool result = false;
  if (!cached) {
    cached = true;
    if (cvars::gpu_debug_markers) {
      result = true;
      XELOGI("GPU debug markers enabled via CVAR");
    } else {
      auto renderdoc_api = xe::ui::RenderDocAPI::CreateIfConnected();
      if (renderdoc_api) {
        result = true;
        XELOGI("GPU debug markers auto-enabled (RenderDoc detected)");
      }
    }
  }
  return result;
}

// TODO(Triang3l): Make accuracy (ROV/FSI) the default when it's optimized
// better (for instance, using static shader modifications to pass render
// target parameters).
DEFINE_string(
    render_target_path, "performance",
    "Render target emulation path to use across all GPU backends.\n"
    "Use: [performance, accuracy]\n"
    " performance:\n"
    "  Host render targets and fixed-function blending and depth/stencil "
    "testing, copying between render targets when needed.\n"
    "  Lower accuracy (limited pixel format support).\n"
    "  Performance limited primarily by render target layout changes requiring "
    "copying, but generally higher.\n"
    "  Maps to 'fbo' on Vulkan and 'rtv' on D3D12.\n"
    " accuracy:\n"
    "  Manual pixel packing, blending and depth/stencil testing, with free "
    "render target layout changes.\n"
    "  Requires GPU supporting fragment shader interlock (Vulkan) or "
    "rasterizer-ordered views (D3D12).\n"
    "  Highest accuracy (all pixel formats handled in software).\n"
    "  Performance limited primarily by overdraw.\n"
    "  Maps to 'fsi' on Vulkan and 'rov' on D3D12.",
    "GPU");

DEFINE_bool(submit_on_primary_buffer_end, true,
            "Submit the command buffer when a PM4 primary buffer ends if it's "
            "possible to submit immediately to try to reduce frame latency.",
            "GPU");

DEFINE_bool(no_discard_stencil_in_transfer_pipelines, false,
            "Skip stencil bit discard in render target transfer pipelines. "
            "May improve performance on some GPUs.",
            "GPU");

DEFINE_bool(
    async_shader_compilation, true,
    "Compile shaders and create pipelines asynchronously in background "
    "threads. "
    "Eliminates shader compilation stutter but may cause brief rendering "
    "artifacts while pipelines are being created. When disabled, pipelines are "
    "created synchronously which causes stutter but no visual artifacts.",
    "GPU");

DEFINE_bool(
    readback_resolve_half_pixel_offset, false,
    "When resolution scaling is active, sample from the center of each scaled "
    "pixel block during readback resolve instead of the top-left corner. May "
    "improve image quality in some cases but can break games that rely on "
    "reading back specific pixel values (e.g., for gamma detection).",
    "GPU");

DEFINE_bool(gpu_3d_to_2d_texture, true,
            "Handle shaders that sample 3D textures as 2D by creating a 2D "
            "texture from slice 0 of the guest memory.",
            "GPU");

DEFINE_int32(anisotropic_override, -1,
             "Forces anisotropic filtering (AF) for eligible textures.\n"
             "Higher values keep textures sharper at oblique angles at the "
             "cost of GPU bandwidth, though most GPUs handle up to 16x fine.\n"
             "In rare cases, forcing AF can introduce visual artifacts.\n"
             " -1 = No override\n"
             "  0 = Disable anisotropic filtering\n"
             "  1 = Force 1x anisotropic filtering\n"
             "  2 = Force 2x anisotropic filtering\n"
             "  3 = Force 4x anisotropic filtering\n"
             "  4 = Force 8x anisotropic filtering\n"
             "  5 = Force 16x anisotropic filtering",
             "GPU");
