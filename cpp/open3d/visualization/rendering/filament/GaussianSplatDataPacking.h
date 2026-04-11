// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Data structures and packing helpers shared between Gaussian splat compute
// backends (Vulkan, Metal).  The structures mirror the std140/std430 layouts
// expected by the GLSL compute shaders.

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cstdint>
#include <cstring>
#include <vector>

#include "open3d/visualization/rendering/Camera.h"
#include "open3d/visualization/rendering/filament/GaussianSplatRenderer.h"

namespace open3d {
namespace visualization {
namespace rendering {

// ----- GPU-compatible structure definitions ----------------------------------
// These must be tightly packed to match the std140/std430 GLSL layouts.

/// vec4 stored as 4 floats (std430 layout).
struct Std430Vec4 {
    float x = 0.f, y = 0.f, z = 0.f, w = 0.f;
};

/// Matches GaussianViewParams uniform block in the compute shaders (std140).
/// Named to match the GLSL block: `layout(std140, binding=0) uniform
/// GaussianViewParams`.
struct alignas(16) GaussianViewParams {
    // mat4 world_from_model (row-major, 4 vec4)
    float world_from_model[16];
    // mat4 view_from_world
    float view_from_world[16];
    // mat4 clip_from_view
    float clip_from_view[16];
    // vec4 camera_position_and_near
    float camera_position_and_near[4];
    // vec4 viewport_origin_and_size
    float viewport_origin_and_size[4];
    // uvec4 scene  (x=splat_count, y=gaussian_splat_sh_degree,
    // z=antialias_flag, w=screen_y_down_flag)
    std::uint32_t scene[4];
    // uvec4 tiles  (xy=tile_size, zw=tile_count)
    std::uint32_t tiles[4];
    // uvec4 limits (x=entry_capacity, y=max_tiles_per_splat,
    // z=max_tile_entries_total, w=tile_key_bits T = ceil(log2(tile_count)),
    // clamped to [1,31]: T bits for tile index, (32-T) bits for depth)
    std::uint32_t limits[4];
    // vec4 depth_range_and_flags (x=near, y=far, z=stable_sort_flag, w=0)
    float depth_range_and_flags[4];
};
static_assert(sizeof(GaussianViewParams) == 288,
              "GaussianViewParams must be 288 bytes to match GLSL layout");

/// Matches the ProjectedGaussian struct in the compute shaders (48 bytes).
/// Named to match the GLSL struct: `struct ProjectedGaussian` in the SSBOs.
/// Layout (std430): vec4(16) + 4×uint(16) + vec4(16) = 48 bytes.
struct alignas(16) ProjectedGaussian {
    Std430Vec4 center_depth_alpha;
    std::uint32_t packed_color;
    std::uint32_t tile_count_overlap;
    std::uint32_t tile_rect_min;
    std::uint32_t tile_rect_max;
    Std430Vec4 inv_basis;
};
static_assert(sizeof(ProjectedGaussian) == 48,
              "ProjectedGaussian must be 48 bytes to match GLSL layout");

/// Matches TileEntry struct in the shaders (4 × uint = 16 bytes).
/// Named to match the GLSL struct: `struct TileEntry` in
/// scatter/sort/composite. stable_index mirrors splat_index; reserved for
/// future stable-sort support (see depth_range_and_flags.z /
/// RenderConfig::stable_sort).
struct TileEntry {
    std::uint32_t depth_key;
    std::uint32_t splat_index;
    std::uint32_t stable_index;  ///< Currently == splat_index; unused until
                                 ///< stable-sort is active.
    std::uint32_t tile_index;  ///< Linear tile index; written by scatter, read
                               ///< by keygen.
};
static_assert(sizeof(TileEntry) == 16,
              "TileEntry must be 16 bytes to match GLSL layout");

/// Global counters written GPU-side (prefix-sum, scatter, dispatch-args) and
/// read CPU-side for diagnostics and error reporting.  Layout matches the
/// GaussianGlobalCounters SSBO block in the compute shaders (std430, binding
/// 10).
struct GaussianGpuCounters {
    std::uint32_t total_entries =
            0;                      ///< Raw total tile entries from prefix-sum.
    std::uint32_t error_flags = 0;  ///< GPU error bitmask (kGaussianGpuError*).
    std::uint32_t tile_count = 0;   ///< Total tile count for the frame.
    std::uint32_t splat_count = 0;  ///< Visible splat count for the frame.
};
static_assert(sizeof(GaussianGpuCounters) == 4 * sizeof(std::uint32_t),
              "GaussianGpuCounters must be 16 bytes");
static constexpr std::size_t kGaussianCounterCount =
        sizeof(GaussianGpuCounters) / sizeof(std::uint32_t);  // 4

// Keep index constants for any code that accesses the buffer as a raw uint
// array.
static constexpr std::size_t kGaussianCounterTotalEntriesIndex = 0;
static constexpr std::size_t kGaussianCounterErrorFlagsIndex = 1;
static constexpr std::size_t kGaussianCounterTileCountIndex = 2;
static constexpr std::size_t kGaussianCounterSplatCountIndex = 3;

inline constexpr std::uint32_t kGaussianGpuErrorTileEntryOverflow = 1u << 0;
inline constexpr std::uint32_t kGaussianGpuErrorSortCountClamped = 1u << 1;
inline constexpr std::uint32_t kGaussianGpuErrorKnownMask =
        kGaussianGpuErrorTileEntryOverflow | kGaussianGpuErrorSortCountClamped;

// ----- CPU-side representation -----------------------------------------------

/// GPU-ready per-splat packed attribute arrays, created once at scene-cache
/// time and uploaded to GPU SSBOs only when the scene content changes.  Stores
/// the same bit formats as the SSBO bindings consumed by gaussian_project.comp:
///   positions       — fp32 vec4, 16 B/splat (fp32 required for position
///   precision) scales          — fp16×4 as uvec2 pair, 8 B/splat; **linear**
///   scale values rotations       — snorm8-biased×4 quat in one uint, 4 B/splat
///   dc_opacity      — fp16×4 as uvec2 pair, 8 B/splat
///   sh_coefficients — fp16 uvec2, stride = 3×degree×2 u32/splat
struct GaussianSplatPackedAttrs {
    std::vector<Std430Vec4> positions;  ///< fp32 vec4, 16 B/splat
    std::vector<std::uint32_t>
            scales;  ///< fp16×4 in uvec2 pair, 8 B/splat; linear
    std::vector<std::uint32_t>
            rotations;  ///< snorm8-biased×4 in uint, 4 B/splat
    std::vector<std::uint32_t> dc_opacity;  ///< fp16×4 in uvec2 pair, 8 B/splat
    std::vector<std::uint32_t>
            sh_coefficients;  ///< fp16 uvec2; degree-dependent stride
    /// Per-splat visibility mask: 1 = render, 0 = cull (via
    /// WriteInvalidProjection). Always present; all-1 when only one object or
    /// all objects are visible.
    std::vector<std::uint32_t> visibility_mask;
    std::uint32_t splat_count = 0;
    int sh_degree = 0;       ///< Effective SH degree packed here
    float min_alpha = 0.0f;  ///< Stored for view-param antialias metadata
    bool antialias = false;
};

/// Per-frame packed view-parameters for GPU upload.
/// Contains only the small per-frame data (UBO + scalar counters).  The large
/// per-splat attribute arrays live in GaussianSplatPackedAttrs (scene
/// lifetime).
struct PackedGaussianScene {
    GaussianViewParams view_params;
    std::uint32_t splat_count = 0;
    std::uint32_t tile_count = 0;
    std::uint32_t pixel_count = 0;
    bool valid = false;
};

// ----- Helper functions used by backends  ------------------------------------

/// Pack camera, viewport, and scene metadata into the view-params UBO.
/// Uses the splat count, SH degree, and antialias flag from `attrs`.
/// Per-frame GPU cost: glBufferSubData / MTLBuffer memcpy of 288 bytes.
PackedGaussianScene PackGaussianViewParams(
        const GaussianSplatPackedAttrs& attrs,
        const GaussianSplatRenderer::ViewRenderData& render_data,
        const GaussianSplatRenderer::RenderConfig& config);

/// Pack Gaussian splat attributes from raw PointCloud data pointers into
/// GPU-ready format, filtering by opacity in a single pass.  Called once at
/// scene cache time (FilamentScene::CacheGaussianSplatData) to eliminate the
/// intermediate fp32 copy that was previously re-packed every scene-change
/// frame.
/// @param n                 Total number of splats in source
/// @param scale_ptr         May be nullptr → zero scales (linear)
/// @param rot_ptr           May be nullptr → identity quaternion
/// @param f_dc_ptr          May be nullptr → zero DC color
/// @param opacity_ptr       May be nullptr → opacity treated as zero
/// @param f_rest_ptr        May be nullptr → no SH rest coefficients
/// @param source_sh_degree  SH degree in the source `f_rest` tensor (sets
/// stride)
/// @param desired_sh_degree Effective degree to pack (may be <
/// source_sh_degree)
/// @param min_opacity_logit Filter threshold in logit space; splats below this
/// are dropped
/// @param min_alpha         Stored as `out.min_alpha` metadata
/// @param antialias         Stored as `out.antialias` metadata
void PackGaussianSplatAttrsDirect(const float* pts_ptr,
                                  std::size_t n,
                                  const float* scale_ptr,
                                  const float* rot_ptr,
                                  const float* f_dc_ptr,
                                  const float* opacity_ptr,
                                  const float* f_rest_ptr,
                                  int source_sh_degree,
                                  int desired_sh_degree,
                                  float min_opacity_logit,
                                  float min_alpha,
                                  bool antialias,
                                  GaussianSplatPackedAttrs& out);

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
