// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
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
struct alignas(16) PackedGaussianViewParams {
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
    // uvec4 scene  (x=splat_count, y=gaussian_splat_sh_degree, z=0, w=0)
    std::uint32_t scene[4];
    // uvec4 tiles  (xy=tile_size, zw=tile_count)
    std::uint32_t tiles[4];
    // vec4 depth_range_and_flags (x=near, y=far, z=stable_sort_flag, w=0)
    float depth_range_and_flags[4];
};

/// Matches the ProjectedGaussian struct in the compute shaders (48 bytes).
/// Layout (std430): vec4(16) + 4×uint(16) + vec4(16) = 48 bytes.
struct alignas(16) PackedProjectedGaussian {
    Std430Vec4 center_depth_alpha;
    std::uint32_t packed_color;
    std::uint32_t tile_count_overlap;
    std::uint32_t tile_rect_min;
    std::uint32_t tile_rect_max;
    Std430Vec4 inv_basis;
};
static_assert(sizeof(PackedProjectedGaussian) == 48,
              "PackedProjectedGaussian must be 48 bytes to match GLSL layout");

/// Matches TileEntry struct in the shaders (4 × uint = 16 bytes).
struct PackedTileEntry {
    std::uint32_t depth_key;
    std::uint32_t splat_index;
    std::uint32_t stable_index;
    std::uint32_t reserved;
};

/// Number of global counters used by the prefix-sum pass.
static constexpr std::size_t kGaussianCounterCount = 4;

// ----- CPU-side packed scene representation ----------------------------------

/// Holds all splat attribute arrays packed for compute upload.
struct PackedGaussianScene {
    PackedGaussianViewParams view_params;
    std::vector<Std430Vec4> positions;
    std::vector<Std430Vec4> log_scales;
    std::vector<Std430Vec4> rotations;
    std::vector<Std430Vec4> dc_opacity;
    std::vector<Std430Vec4> sh_coefficients;
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
};

// ----- Helper functions used by backends  ------------------------------------

/// Pack camera, viewport, and scene data into a PackedGaussianScene.
PackedGaussianScene PackGaussianSceneInputs(
        const GaussianSplatSourceData& source,
        const GaussianComputeRenderer::ViewRenderData& render_data,
        const GaussianComputeRenderer::RenderConfig& config);

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
