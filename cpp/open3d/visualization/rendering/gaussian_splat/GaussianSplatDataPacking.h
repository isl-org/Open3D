// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// CPU data structures and packing helpers shared by the OpenGL and Metal
// Gaussian splat compute backends.  The structures mirror the std140/std430
// layouts expected by the GLSL/MSL compute shaders.

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cstdint>
#include <cstring>
#include <vector>

#include "open3d/visualization/rendering/Camera.h"
#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatRenderer.h"

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
    // vec4 depth_range_and_flags (x=near, y=far, z=reserved, w=0)
    float depth_range_and_flags[4];
};
static_assert(sizeof(GaussianViewParams) == 288,
              "GaussianViewParams must be 288 bytes to match GLSL layout");

/// Composite-pass projected splat data (32 bytes, binding 6).
/// Written by gaussian_project.comp, read only by gaussian_composite.comp.
/// Layout (std430): 8×uint/float = 32 bytes (two vec4).
///   center_xy_fp16 — packHalf2x16(center_x, center_y) in absolute viewport
///                    pixels; fp16 step ≤ 0.5 px at 4K.
///   depth          — normalized linear depth (near=0, far=1), fp32.
///   alpha          — sigmoid(opacity) × density_compensation, fp32.
///   packed_rgba8 — RGBA8-packed view-dependent SH color.
///   inv_basis    — vec4: 2×2 inverse covariance basis (row-major), std430-
///                  aligned to 16 B; composite reads 32 B from one buffer.
struct alignas(16) ProjectedComposite {
    std::uint32_t center_xy_fp16;
    float depth;
    float alpha;
    std::uint32_t packed_rgba8;
    alignas(16) Std430Vec4 inv_basis;
};
static_assert(sizeof(ProjectedComposite) == 32,
              "ProjectedComposite must be 32 bytes to match GLSL layout");

/// Tile-metadata projected splat data (16 bytes, binding 12).
/// Written by gaussian_project.comp, read by gaussian_prefix_sum.comp and
/// gaussian_scatter.comp (never by the composite pass).
/// Layout (std430): 1×float + 3×uint = 16 bytes.
///   norm_depth         — copy of depth; used as the sort key source.
///   tile_count_overlap — number of tiles the splat overlaps; 0 = culled.
///   tile_rect_min/max  — packed tile bounding box: (y<<16)|x.
struct alignas(16) ProjectedTileMeta {
    float norm_depth;
    std::uint32_t tile_count_overlap;
    std::uint32_t tile_rect_min;
    std::uint32_t tile_rect_max;
};
static_assert(sizeof(ProjectedTileMeta) == 16,
              "ProjectedTileMeta must be 16 bytes to match GLSL layout");

/// Matches TileEntry struct in the shaders (3 × uint = 12 bytes).
/// Named to match the GLSL struct: `struct TileEntry` in
/// scatter/sort/composite. Radix sort is inherently stable so stable_index
/// is not needed.
struct TileEntry {
    std::uint32_t depth_key;
    std::uint32_t splat_index;
    std::uint32_t tile_index;  ///< Linear tile index; written by scatter, read
                               ///< by keygen.
};
static_assert(sizeof(TileEntry) == 12,
              "TileEntry must be 12 bytes to match GLSL layout");

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

// Index constants for code that accesses the counters buffer as a raw uint32
// array rather than through the struct.
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
    /// Bit-packed visibility mask: bit k = 1 → render splat k, 0 → cull (via
    /// WriteInvalidProjection). Size = ceil(splat_count / 32) uint32 words.
    /// All bits set when all splats are visible.
    std::vector<std::uint32_t> visibility_mask;
    std::uint32_t splat_count = 0;
    int sh_degree = 0;  ///< Effective SH degree packed here
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

// ----- GPU buffer sizing (absorbed from GaussianSplatBuffers) ----------------

/// UBO stride for radix-sort params: must match
/// GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT padding used by both the OpenGL backend
/// and the radix-sort dispatch shaders.
inline constexpr std::uint32_t kGaussianRadixParamsStride = 256;

/// Number of circular partition-buffer slots in the OneSweep sort.
/// Must be a power of 2 and exceed the maximum number of simultaneously
/// live workgroups (typically ≤ 384 on Intel Iris Xe). Must match CIRCULAR_SIZE
/// in gaussian_onesweep_digit_pass_subgroup.comp.
/// Fixed buffer = kOneSweepCircularSize × 256 × 8 B = 1 MB.
inline constexpr std::uint32_t kOneSweepCircularSize = 512u;

/// Byte sizes and capacities for Gaussian splat SSBOs/UBOs.
/// Must match allocation logic in gaussian_compute_dispatch_args.comp and the
/// radix shaders (tile cap, etc.).
struct GaussianGpuBufferSizes {
    /// Two projected buffers: composite (32 B, composite pass only) and
    /// tile metadata (16 B, scatter/prefix passes only).
    std::size_t projected_composite_size = 0;  ///< splat_count × 32 B
    std::size_t projected_meta_size = 0;       ///< splat_count × 16 B
    std::size_t tile_scalar_size = 0;
    std::size_t entry_buf_size = 0;
    std::size_t key_cap_size = 0;
    /// Byte size of the sorted splat-index buffer: one uint32 per tile entry
    /// (4 B each, vs. 12 B for a full TileEntry). Used by the composite pass.
    std::size_t sorted_splat_size = 0;
    std::size_t histogram_buf_size = 0;
    /// 10 radix dispatch entries + 5 OneSweep dispatch entries (always
    /// emitted).
    std::size_t dispatch_args_size = 0;
    std::size_t radix_params_size = 0;
    std::uint32_t entries_capacity = 0;
    std::uint32_t radix_num_wg_cap = 0;
    /// OneSweep sort buffers (allocated only when use_onesweep_sort=true):
    std::size_t onesweep_global_hist_size =
            0;  ///< 4×256×4 = 4 KB, always fixed
    /// Fixed 1 MB: kOneSweepCircularSize × 256 × 8 B (uvec2 per slot).
    /// Does not grow with sort size; stays hot in iGPU LLC.
    std::size_t onesweep_partition_size = 0;
    std::size_t onesweep_tail_size = 0;  ///< 1 uint32 (4 B)
};

/// Compute GPU buffer sizes from a packed-scene frame description.
/// Called once per geometry pass to size intermediate SSBO allocations.
void ComputeGaussianGpuBufferSizes(const PackedGaussianScene& packed,
                                   GaussianGpuBufferSizes* out);

// ----- Packing helpers -------------------------------------------------------

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
/// @param scale_ptr         Non-null when n>0: linear scales (3 floats/splat).
/// @param rot_ptr           Non-null when n>0: unit quaternion (w,x,y,z).
/// @param f_dc_ptr          Non-null when n>0: DC SH color (3 floats/splat).
/// @param opacity_ptr       Non-null when n>0: opacity logit (1 float/splat).
/// @param f_rest_ptr        May be nullptr → no SH rest coefficients
/// @param source_sh_degree  SH degree in the source `f_rest` tensor (sets
/// stride)
/// @param desired_sh_degree Effective degree to pack (may be <
/// source_sh_degree)
/// @param min_opacity_logit Filter threshold in logit space; splats below this
/// are dropped
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
                                  bool antialias,
                                  GaussianSplatPackedAttrs& out);

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
