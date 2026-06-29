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

// fp16-pair storage stride in uint32 words for one splat's SH-rest payload.
// Degree-0 has no f_rest payload.
std::size_t GetShU32PerSplat(int sh_degree) {
    return sh_degree > 0 ? std::size_t(3 * sh_degree * 2) : 0u;
}

void ReservePackedAttrArrays(GaussianSplatPackedAttrs* out,
                             std::size_t splat_capacity,
                             std::size_t sh_u32_per_splat) {
    out->positions.reserve(splat_capacity);
    out->scales.reserve(2 * splat_capacity);
    out->rotations.reserve(splat_capacity);
    out->dc_opacity.reserve(2 * splat_capacity);
    if (sh_u32_per_splat > 0u) {
        out->sh_coefficients.reserve(splat_capacity * sh_u32_per_splat);
    }
}

std::size_t AppendZeroedShSlot(std::vector<std::uint32_t>* dst_sh,
                               std::size_t sh_u32_per_splat) {
    const std::size_t dst_base = dst_sh->size();
    dst_sh->resize(dst_base + sh_u32_per_splat, 0u);
    return dst_base;
}

void CopyShPrefixToSlot(std::vector<std::uint32_t>* dst_sh,
                        std::size_t dst_base,
                        std::size_t dst_sh_u32_per_splat,
                        const std::uint32_t* src_u32,
                        std::size_t src_u32_count) {
    const std::size_t copy_count =
            std::min(dst_sh_u32_per_splat, src_u32_count);
    if (copy_count == 0u) {
        return;
    }
    std::copy_n(src_u32, copy_count, dst_sh->begin() + dst_base);
}

void SetVisibilityMaskRange(std::vector<std::uint32_t>* visibility_mask,
                            std::uint32_t splat_start,
                            std::uint32_t splat_count,
                            bool visible) {
    const std::uint32_t splat_end = splat_start + splat_count;
    const std::uint32_t n_words = (splat_end + 31u) / 32u;
    visibility_mask->resize(n_words, 0u);
    if (!visible) {
        return;
    }

    for (std::uint32_t k = splat_start; k < splat_end; ++k) {
        (*visibility_mask)[k >> 5u] |= 1u << (k & 31u);
    }
}

// Pack fp32 → fp16 bit patterns. Matches GLSL unpackHalf2x16() (lo bits 0–15,
// hi bits 16–31). Use Eigen's conversion helpers instead of reading
// Eigen::half::x directly because on ARM64 that field is __fp16 rather than the
// raw 16-bit bit pattern.
inline std::uint16_t FloatToHalfBits(float value) {
    return Eigen::half_impl::raw_half_as_uint16(
            Eigen::half_impl::float_to_half_rtne(value));
}

inline std::uint32_t HalfPair(float lo, float hi) {
    return static_cast<std::uint32_t>(FloatToHalfBits(lo)) |
           (static_cast<std::uint32_t>(FloatToHalfBits(hi)) << 16u);
}

// Four floats → two u32 (two half2), e.g. linear scales (sx,sy,sz,0) or
// DC+opacity (r,g,b,sigmoid(opacity_logit)).
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
    // ProjectedComposite (32 B): written by project, read only by composite.
    out->projected_composite_size =
            packed.splat_count * sizeof(ProjectedComposite);
    // steal_counter: one uint32 — atomic tile index for composite
    // work-stealing. Composite no longer needs per-tile counts/offsets arrays
    // (binary search).
    out->tile_scalar_size = sizeof(std::uint32_t);

    out->entries_capacity =
            std::max<std::uint32_t>(1u, packed.view_params.limits[0]);
    // Sort key/value ping-pong buffers: one uint32 per entry capacity.
    out->key_cap_size = std::max<std::size_t>(
            4u, static_cast<std::size_t>(out->entries_capacity) *
                        sizeof(std::uint32_t));

    out->radix_num_wg_cap = std::max(
            1u, (out->entries_capacity +
                 kRadixWorkgroupSize * kRadixTargetBlocksPerWG - 1u) /
                        (kRadixWorkgroupSize * kRadixTargetBlocksPerWG));
    out->histogram_buf_size = std::max<std::size_t>(
            4u, static_cast<std::size_t>(out->radix_num_wg_cap) *
                        kRadixSortBins * sizeof(std::uint32_t));

    // 8 slots: 4×histogram + 4×radix_scatter.
    out->dispatch_args_size = 8u * 3u * sizeof(std::uint32_t);
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
    std::size_t sh_u32_per_splat = 0;
    if (desired_sh_degree > 0 && f_rest_ptr) {
        source_coeffs = GetGaussianSplatRestCoeffCount(source_sh_degree);
        desired_coeffs = GetGaussianSplatRestCoeffCount(desired_sh_degree);
        sh_u32_per_splat = GetShU32PerSplat(desired_sh_degree);
    }

    ReservePackedAttrArrays(&out, n, sh_u32_per_splat);

    for (std::size_t i = 0; i < n; ++i) {
        const float opacity = opacity_ptr[i];
        if (opacity < min_opacity_logit) {
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

        // Apply sigmoid to the raw opacity logit once at pack time; the shader
        // reads the result directly as alpha without a per-frame exp() call.
        const float sigmoid_opacity = 1.0f / (1.0f + std::exp(-opacity));
        const auto dc = PackHalf4(f_dc_ptr[i * 3 + 0], f_dc_ptr[i * 3 + 1],
                                  f_dc_ptr[i * 3 + 2], sigmoid_opacity);
        out.dc_opacity.insert(out.dc_opacity.end(), dc.begin(), dc.end());

        if (sh_u32_per_splat > 0 && f_rest_ptr) {
            const std::size_t dst_base =
                    AppendZeroedShSlot(&out.sh_coefficients, sh_u32_per_splat);
            const float* src = f_rest_ptr + i * source_coeffs;
            const int pairs = desired_coeffs / 2;
            for (int p = 0; p < pairs; ++p) {
                out.sh_coefficients[dst_base + std::size_t(p)] =
                        HalfPair(src[p * 2 + 0], src[p * 2 + 1]);
            }
            if (desired_coeffs & 1) {
                out.sh_coefficients[dst_base + std::size_t(pairs)] =
                        HalfPair(src[desired_coeffs - 1], 0.f);
            }
        }
    }

    out.splat_count = static_cast<std::uint32_t>(out.positions.size());
    // Populate a default all-visible mask so the buffer is always present.
    // RebuildMergedGaussianData() overwrites this with per-object visibility.
    // Bit-packed: all-ones = all splats visible. ceil(n/32) words.
    out.visibility_mask.clear();
    SetVisibilityMaskRange(&out.visibility_mask, 0u, out.splat_count, true);
}

void MergeGaussianSplatPackedAttrs(
        const std::vector<GaussianSplatMergeItem>& items,
        GaussianSplatPackedAttrs* out,
        std::vector<std::uint32_t>* splat_starts) {
    if (!out) {
        return;
    }

    *out = GaussianSplatPackedAttrs{};
    if (splat_starts) {
        splat_starts->clear();
        splat_starts->reserve(items.size());
    }

    std::uint32_t merged_sh_degree = 0u;
    std::size_t merged_splat_capacity = 0u;
    for (const auto& item : items) {
        if (!item.attrs) {
            continue;
        }
        merged_sh_degree = std::max(merged_sh_degree,
                                    std::uint32_t(item.attrs->sh_degree));
        merged_splat_capacity += item.attrs->splat_count;
    }

    out->sh_degree = static_cast<int>(merged_sh_degree);
    const std::size_t merged_sh_u32_per_splat =
            GetShU32PerSplat(static_cast<int>(merged_sh_degree));
    ReservePackedAttrArrays(out, merged_splat_capacity,
                            merged_sh_u32_per_splat);

    for (const auto& item : items) {
        if (splat_starts) splat_starts->push_back(out->splat_count);
        if (!item.attrs) {
            continue;
        }
        const auto& src = *item.attrs;
        const std::uint32_t splat_count = src.splat_count;

        out->positions.insert(out->positions.end(), src.positions.begin(),
                              src.positions.end());
        out->scales.insert(out->scales.end(), src.scales.begin(),
                           src.scales.end());
        out->rotations.insert(out->rotations.end(), src.rotations.begin(),
                              src.rotations.end());
        out->dc_opacity.insert(out->dc_opacity.end(), src.dc_opacity.begin(),
                               src.dc_opacity.end());

        if (merged_sh_u32_per_splat > 0u) {
            const std::size_t src_sh_u32_per_splat =
                    GetShU32PerSplat(src.sh_degree);
            const std::size_t src_sh_size = src.sh_coefficients.size();
            for (std::uint32_t i = 0; i < splat_count; ++i) {
                const std::size_t dst_base = AppendZeroedShSlot(
                        &out->sh_coefficients, merged_sh_u32_per_splat);
                if (src_sh_u32_per_splat == 0u) {
                    continue;
                }
                const std::size_t src_base =
                        std::size_t(i) * src_sh_u32_per_splat;
                if (src_base >= src_sh_size) {
                    continue;
                }
                CopyShPrefixToSlot(
                        &out->sh_coefficients, dst_base,
                        merged_sh_u32_per_splat,
                        src.sh_coefficients.data() + src_base,
                        std::min(src_sh_u32_per_splat, src_sh_size - src_base));
            }
        }

        SetVisibilityMaskRange(&out->visibility_mask, out->splat_count,
                               splat_count, item.visible);
        out->splat_count += splat_count;
        out->antialias = out->antialias || src.antialias;
    }
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
