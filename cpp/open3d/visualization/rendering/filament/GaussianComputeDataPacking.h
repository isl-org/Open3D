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
#include "open3d/visualization/rendering/filament/GaussianComputeRenderer.h"

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
/// Named to match the GLSL block: `layout(std140, binding=0) uniform GaussianViewParams`.
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
/// Named to match the GLSL struct: `struct TileEntry` in scatter/sort/composite.
/// stable_index mirrors splat_index; reserved for future stable-sort support
/// (see depth_range_and_flags.z / RenderConfig::stable_sort).
struct TileEntry {
    std::uint32_t depth_key;
    std::uint32_t splat_index;
    std::uint32_t stable_index;  ///< Currently == splat_index; unused until stable-sort is active.
    std::uint32_t tile_index;    ///< Linear tile index; written by scatter, read by keygen.
};
static_assert(sizeof(TileEntry) == 16,
              "TileEntry must be 16 bytes to match GLSL layout");

/// Global counters written GPU-side (prefix-sum, scatter, dispatch-args) and
/// read CPU-side for diagnostics and error reporting.  Layout matches the
/// GaussianGlobalCounters SSBO block in the compute shaders (std430, binding 10).
struct GaussianGpuCounters {
    std::uint32_t total_entries = 0;  ///< Raw total tile entries from prefix-sum.
    std::uint32_t error_flags   = 0;  ///< GPU error bitmask (kGaussianGpuError*).
    std::uint32_t tile_count    = 0;  ///< Total tile count for the frame.
    std::uint32_t splat_count   = 0;  ///< Visible splat count for the frame.
};
static_assert(sizeof(GaussianGpuCounters) == 4 * sizeof(std::uint32_t),
              "GaussianGpuCounters must be 16 bytes");
static constexpr std::size_t kGaussianCounterCount =
        sizeof(GaussianGpuCounters) / sizeof(std::uint32_t);  // 4

// Keep index constants for any code that accesses the buffer as a raw uint array.
static constexpr std::size_t kGaussianCounterTotalEntriesIndex = 0;
static constexpr std::size_t kGaussianCounterErrorFlagsIndex   = 1;
static constexpr std::size_t kGaussianCounterTileCountIndex    = 2;
static constexpr std::size_t kGaussianCounterSplatCountIndex   = 3;

inline constexpr std::uint32_t kGaussianGpuErrorTileEntryOverflow = 1u << 0;
inline constexpr std::uint32_t kGaussianGpuErrorSortCountClamped = 1u << 1;
inline constexpr std::uint32_t kGaussianGpuErrorKnownMask =
    kGaussianGpuErrorTileEntryOverflow |
    kGaussianGpuErrorSortCountClamped;

// ----- CPU-side packed scene representation ----------------------------------

/// Holds all splat attribute arrays packed for compute upload.
/// Field formats match the SSBO layout in gaussian_project.comp:
///   positions        — fp32 vec4, 16 B/splat
///   log_scales       — fp16×4 as uvec2 pair, 8 B/splat
///   rotations        — snorm8-biased×4 quat in one uint, 4 B/splat
///   dc_opacity       — fp16×4 as uvec2 pair, 8 B/splat
///   sh_coefficients  — fp16 uvec2, stride=3×degree (0/24/48 B/splat)
struct PackedGaussianScene {
    GaussianViewParams view_params;
    std::vector<Std430Vec4> positions;      ///< fp32 vec4, 16 B/splat
    std::vector<std::uint32_t> log_scales;  ///< fp16×4 in uvec2 pair, 8 B/splat
    std::vector<std::uint32_t>
            rotations;  ///< snorm8-biased×4 in uint, 4 B/splat
    std::vector<std::uint32_t> dc_opacity;  ///< fp16×4 in uvec2 pair, 8 B/splat
    std::vector<std::uint32_t>
            sh_coefficients;  ///< fp16 uvec2; 0/6/12 u32/splat (deg 0/1/2)
    std::uint32_t splat_count = 0;
    std::uint32_t tile_count = 0;
    std::uint32_t pixel_count = 0;
    bool valid = false;
};

/// CPU-side cached copy of Gaussian splat source attributes.
/// Stored alongside the scene so that compute backends can pack input buffers
/// without reading back from Filament vertex buffers.
struct GaussianSplatSourceData {
    std::vector<float> positions;   ///< 3 floats per splat (x, y, z)
    std::vector<float> log_scales;  ///< 3 floats per splat
    std::vector<float> rotations;   ///< 4 floats per splat (quat w, x, y, z)
    std::vector<float> dc_opacity;  ///< 4 floats per splat (r, g, b, opacity)
    std::vector<float> sh_rest;     ///< Variable length SH coefficients
    std::uint32_t splat_count = 0;
    int gaussian_splat_sh_degree = 2;
    float gaussian_splat_min_alpha = 0.0f;
    bool gaussian_splat_antialias = false;
};

// ----- Helper functions used by backends  ------------------------------------

/// Pack camera, viewport, and scene metadata into the view-params UBO and the
/// scalar fields (splat_count, tile_count, pixel_count) of a
/// PackedGaussianScene.  The per-splat attribute vectors (positions, rotations,
/// scales, SH) are intentionally left empty — call PackGaussianSceneAttributes
/// separately when the scene changes.  Per-frame GPU cost: glBufferSubData /
/// MTLBuffer memcpy of 288 bytes (one GaussianViewParams).
PackedGaussianScene PackGaussianViewParams(
        const GaussianSplatSourceData& source,
        const GaussianComputeRenderer::ViewRenderData& render_data,
        const GaussianComputeRenderer::RenderConfig& config);

/// Fill the large per-splat geometry attribute arrays (positions, scales,
/// rotations, dc_opacity, SH coefficients) in a PackedGaussianScene
/// previously created by PackGaussianViewParams.  For N splats with degree-2
/// SH this allocates and writes N * ~160 bytes; only call when the splat
/// geometry (splat count or content) actually changes.
void PackGaussianSceneAttributes(
        const GaussianSplatSourceData& source,
        const GaussianComputeRenderer::RenderConfig& config,
        PackedGaussianScene& packed);

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
