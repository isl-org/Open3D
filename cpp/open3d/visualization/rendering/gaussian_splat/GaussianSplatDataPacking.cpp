// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// CPU-side packing and GPU buffer-sizing implementation for the Gaussian splat
// compute pipeline (OpenGL and Metal backends share this code).

#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatDataPacking.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

namespace open3d {
namespace visualization {
namespace rendering {

namespace {

// Integer ceiling division: ceil(value / divisor).
// Used locally to compute tile grid dimensions.
inline int CeilDiv(int value, int divisor) {
    return (value + divisor - 1) / divisor;
}

// Pack an Eigen 4x4 (column-major) into a 16-float array in column-major order
// matching std140 mat4 layout.
void PackMat4(float* dst, const Eigen::Matrix4f& m) {
    std::memcpy(dst, m.data(), 16 * sizeof(float));
}

// Count rest-coefficient floats per splat for a given SH degree.
// Degree-0 (DC) lives in dc_opacity; sh_rest stores degrees 1..sh_degree,
// i.e. ((degree+1)^2 - 1) basis functions per RGB channel.
int GetGaussianSplatRestCoeffCount(int sh_degree) {
    return (((sh_degree + 1) * (sh_degree + 1)) - 1) * 3;
}

// Pack fp32 → fp16 bit patterns. Matches GLSL unpackHalf2x16() (lo bits 0–15,
// hi bits 16–31). Eigen::half uses round-to-nearest-even; .x is raw uint16.
inline std::uint32_t HalfPair(float lo, float hi) {
    return static_cast<std::uint32_t>(Eigen::half(lo).x) |
           (static_cast<std::uint32_t>(Eigen::half(hi).x) << 16u);
}

// Four floats → two u32 (two half2), e.g. linear scales (sx,sy,sz,0) or
// DC+opacity (r,g,b,opacity_logit).
inline std::array<std::uint32_t, 2> PackHalf4(float x,
                                              float y,
                                              float z,
                                              float w) {
    return {{HalfPair(x, y), HalfPair(z, w)}};
}

// Encode a float in [-1, 1] as a biased uint8 (stored value =
// int8(round(f*127)) + 128). GPU decode: (float(byte) - 128.0) / 127.0.
std::uint8_t EncodeSnorm8(float f) {
    const float clamped = std::max(-1.0f, std::min(1.0f, f));
    const auto q = static_cast<std::int8_t>(
            static_cast<int>(std::round(clamped * 127.0f)));
    return static_cast<std::uint8_t>(static_cast<int>(q) + 128);
}

// Pack quaternion components (w, x, y, z) as 4 biased snorm8 values into one
// uint32 (w in bits 0-7, x in 8-15, y in 16-23, z in 24-31).
std::uint32_t PackSnorm8x4(float w, float x, float y, float z) {
    return static_cast<std::uint32_t>(EncodeSnorm8(w)) |
           (static_cast<std::uint32_t>(EncodeSnorm8(x)) << 8) |
           (static_cast<std::uint32_t>(EncodeSnorm8(y)) << 16) |
           (static_cast<std::uint32_t>(EncodeSnorm8(z)) << 24);
}

// Radix-sort workgroup constants — must stay in sync with the radix shaders
// and the dispatch-args compute shader.
constexpr std::uint32_t kRadixWorkgroupSize = 256;
constexpr std::uint32_t kRadixSortBins = 256;
constexpr std::uint32_t kRadixTargetBlocksPerWG = 32;

}  // namespace

// ----- ComputeGaussianGpuBufferSizes -----------------------------------------

void ComputeGaussianGpuBufferSizes(const PackedGaussianScene& packed,
                                   GaussianGpuBufferSizes* out) {
    // Derive SSBO/UBO byte sizes from the packed-scene tile/splat counts.
    // All size formulas must stay in sync with the dispatch shaders.
    if (!out || !packed.valid) {
        return;
    }
    // ProjectedComposite (32 B): Std430Vec4 inv_basis matches GLSL vec4.
    // so one buffer covers all composite-pass per-splat data.
    out->projected_composite_size =
            packed.splat_count * sizeof(ProjectedComposite);
    out->projected_meta_size = packed.splat_count * sizeof(ProjectedTileMeta);
    out->tile_scalar_size = packed.tile_count * sizeof(std::uint32_t);

    out->entries_capacity =
            std::max<std::uint32_t>(1u, packed.view_params.limits[0]);
    out->entry_buf_size = std::max(
            sizeof(TileEntry), static_cast<std::size_t>(out->entries_capacity) *
                                       sizeof(TileEntry));

    out->key_cap_size = std::max<std::size_t>(
            4u, static_cast<std::size_t>(out->entries_capacity) *
                        sizeof(std::uint32_t));
    // Sorted splat-index buffer: one uint32 per entry (3× smaller than a full
    // TileEntry). key_cap_size is already one uint32 per entry — reuse it.
    out->sorted_splat_size = out->key_cap_size;

    out->radix_num_wg_cap = std::max(
            1u, (out->entries_capacity +
                 kRadixWorkgroupSize * kRadixTargetBlocksPerWG - 1u) /
                        (kRadixWorkgroupSize * kRadixTargetBlocksPerWG));
    out->histogram_buf_size = std::max<std::size_t>(
            4u, static_cast<std::size_t>(out->radix_num_wg_cap) *
                        kRadixSortBins * sizeof(std::uint32_t));

    out->dispatch_args_size = 10u * 3u * sizeof(std::uint32_t);
    out->radix_params_size = 4u * kGaussianRadixParamsStride;
}

// ----- PackGaussianViewParams ------------------------------------------------

PackedGaussianScene PackGaussianViewParams(
        const GaussianSplatPackedAttrs& attrs,
        const GaussianSplatRenderer::ViewRenderData& render_data,
        const GaussianSplatRenderer::RenderConfig& config) {
    // Build the 288-byte per-frame UBO from camera/viewport/scene state.
    // Called every frame; cost is a single memcpy-sized CPU write to the GPU.
    PackedGaussianScene packed;

    if (attrs.splat_count == 0 || render_data.viewport_size.x() <= 0 ||
        render_data.viewport_size.y() <= 0) {
        return packed;
    }

    const std::uint32_t n = attrs.splat_count;
    const int w = render_data.viewport_size.x();
    const int h = render_data.viewport_size.y();
    const int tx = config.tile_size.x();
    const int ty = config.tile_size.y();
    const int tcx = CeilDiv(w, tx);
    const int tcy = CeilDiv(h, ty);

    auto& vp = packed.view_params;
    std::memset(&vp, 0, sizeof(vp));

    const auto entry_budget =
            static_cast<std::uint64_t>(n) * config.max_tiles_per_splat;
    const auto entry_capacity = static_cast<std::uint32_t>(std::min(
            entry_budget,
            static_cast<std::uint64_t>(config.max_tile_entries_total)));

    // world_from_model: splat positions are in world space → identity.
    // render_data.model_matrix is the camera rig; do not use it here.
    PackMat4(vp.world_from_model, Eigen::Matrix4f::Identity());
    PackMat4(vp.view_from_world,
             render_data.view_matrix.matrix().cast<float>());
    PackMat4(vp.clip_from_view, render_data.projection_matrix.matrix());

    const std::array<float, 4> cam_pn{
            render_data.camera_position.x(), render_data.camera_position.y(),
            render_data.camera_position.z(),
            static_cast<float>(render_data.near_plane)};
    std::memcpy(vp.camera_position_and_near, cam_pn.data(), sizeof(cam_pn));

    const std::array<float, 4> vp_origin_size{
            static_cast<float>(render_data.viewport_origin.x()),
            static_cast<float>(render_data.viewport_origin.y()),
            static_cast<float>(w), static_cast<float>(h)};
    std::memcpy(vp.viewport_origin_and_size, vp_origin_size.data(),
                sizeof(vp_origin_size));

    const int effective_sh_degree =
            std::min(attrs.sh_degree, config.max_sh_degree);
    const std::array<std::uint32_t, 4> scene_u{
            n, static_cast<std::uint32_t>(effective_sh_degree),
            (attrs.antialias || config.antialias) ? 1u : 0u,
            render_data.screen_y_down ? 1u : 0u};
    std::memcpy(vp.scene, scene_u.data(), sizeof(scene_u));

    const std::array<std::uint32_t, 4> tiles_u{
            static_cast<std::uint32_t>(tx), static_cast<std::uint32_t>(ty),
            static_cast<std::uint32_t>(tcx), static_cast<std::uint32_t>(tcy)};
    std::memcpy(vp.tiles, tiles_u.data(), sizeof(tiles_u));

    vp.limits[0] = entry_capacity;
    vp.limits[1] = config.max_tiles_per_splat;
    vp.limits[2] = config.max_tile_entries_total;
    // T = bits needed to hold the largest tile index =
    // floor(log2(max_index))+1. For a single tile (tcx*tcy == 1) max_tile_index
    // == 0, so T defaults to 1 (no zero-shift UB). Clamped to [1,31] so the
    // depth field gets >= 1 bit.
    const std::uint32_t max_tile_index =
            static_cast<std::uint32_t>(tcx * tcy) - 1u;
    std::uint32_t tile_key_bits = 0u;
    for (std::uint32_t v = max_tile_index; v > 0u; v >>= 1u) {
        ++tile_key_bits;
    }
    vp.limits[3] = std::clamp(tile_key_bits, 1u, 31u);

    const std::array<float, 4> depth_rng{
            static_cast<float>(render_data.near_plane),
            static_cast<float>(render_data.far_plane), 0.f, 0.f};
    std::memcpy(vp.depth_range_and_flags, depth_rng.data(), sizeof(depth_rng));

    packed.splat_count = n;
    packed.tile_count = static_cast<std::uint32_t>(tcx * tcy);
    packed.pixel_count = static_cast<std::uint32_t>(w * h);
    packed.valid = true;
    return packed;
}

// ----- PackGaussianSplatAttrsDirect ------------------------------------------

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
                                  GaussianSplatPackedAttrs& out) {
    // Pack all per-splat attributes into GPU-ready fp16/snorm8 arrays in a
    // single pass, filtering below-threshold splats to keep upload small.
    // scale_ptr, rot_ptr, f_dc_ptr, opacity_ptr must be non-null when n>0
    // (FilamentScene rejects clouds missing any of these). f_rest_ptr is
    // optional. For n==0, pointers are unused (empty std::vector::data() may be
    // nullptr).
    out = GaussianSplatPackedAttrs{};
    out.sh_degree = desired_sh_degree;
    out.antialias = antialias;
    if (n == 0) {
        out.visibility_mask.assign(0, 0u);
        return;
    }
    if (!pts_ptr || !scale_ptr || !rot_ptr || !f_dc_ptr || !opacity_ptr) {
        return;
    }

    int source_coeffs = 0;
    int desired_coeffs = 0;
    int sh_u32_per_splat = 0;
    if (desired_sh_degree > 0 && f_rest_ptr) {
        source_coeffs = GetGaussianSplatRestCoeffCount(source_sh_degree);
        desired_coeffs = GetGaussianSplatRestCoeffCount(desired_sh_degree);
        sh_u32_per_splat = 3 * desired_sh_degree * 2;
    }

    out.positions.reserve(n);
    out.scales.reserve(2 * n);
    out.rotations.reserve(n);
    out.dc_opacity.reserve(2 * n);
    if (sh_u32_per_splat > 0) {
        out.sh_coefficients.reserve(static_cast<std::size_t>(sh_u32_per_splat) *
                                    n);
    }

    for (std::size_t i = 0; i < n; ++i) {
        if (const float opacity = opacity_ptr[i]; opacity < min_opacity_logit) {
            continue;
        }

        out.positions.push_back({pts_ptr[i * 3 + 0], pts_ptr[i * 3 + 1],
                                 pts_ptr[i * 3 + 2], 0.f});

        const auto sc = PackHalf4(scale_ptr[i * 3 + 0], scale_ptr[i * 3 + 1],
                                  scale_ptr[i * 3 + 2], 0.f);
        out.scales.insert(out.scales.end(), sc.begin(), sc.end());

        out.rotations.push_back(
                PackSnorm8x4(rot_ptr[i * 4 + 0], rot_ptr[i * 4 + 1],
                             rot_ptr[i * 4 + 2], rot_ptr[i * 4 + 3]));

        const auto dc = PackHalf4(f_dc_ptr[i * 3 + 0], f_dc_ptr[i * 3 + 1],
                                  f_dc_ptr[i * 3 + 2], opacity_ptr[i]);
        out.dc_opacity.insert(out.dc_opacity.end(), dc.begin(), dc.end());

        if (sh_u32_per_splat > 0 && f_rest_ptr) {
            const float* src = f_rest_ptr + i * source_coeffs;
            const int pairs = desired_coeffs / 2;
            for (int p = 0; p < pairs; ++p) {
                out.sh_coefficients.push_back(
                        HalfPair(src[p * 2 + 0], src[p * 2 + 1]));
            }
            if (desired_coeffs & 1) {
                out.sh_coefficients.push_back(
                        HalfPair(src[desired_coeffs - 1], 0.f));
            }
            const int packed_u32s = (desired_coeffs + 1) / 2;
            for (int p = packed_u32s; p < sh_u32_per_splat; ++p) {
                out.sh_coefficients.push_back(0u);
            }
        }
    }

    out.splat_count = static_cast<std::uint32_t>(out.positions.size());
    // Populate a default all-visible mask so the buffer is always present.
    // RebuildMergedGaussianData() overwrites this with per-object visibility.
    // Bit-packed: all-ones = all splats visible. ceil(n/32) words.
    const std::uint32_t n_mask_words = (out.splat_count + 31u) / 32u;
    out.visibility_mask.assign(n_mask_words, ~0u);
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
