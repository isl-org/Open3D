// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Implementation of data packing helpers for the Gaussian splat compute
// renderer.

#include "open3d/visualization/rendering/filament/GaussianSplatDataPacking.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "open3d/visualization/rendering/filament/GaussianSplatUtils.h"

namespace open3d {
namespace visualization {
namespace rendering {

namespace {

// Pack an Eigen 4x4 (column-major) into a 16-float array in column-major order
// matching std140 mat4 layout.
void PackMat4(float* dst, const Eigen::Matrix4f& m) {
    std::memcpy(dst, m.data(), 16 * sizeof(float));
}

int GetGaussianSplatRestCoeffCount(int sh_degree) {
    // Degree-0 (DC) lives in dc_opacity. sh_rest stores only degrees
    // 1..sh_degree, i.e. ((degree + 1)^2 - 1) basis functions per channel.
    return (((sh_degree + 1) * (sh_degree + 1)) - 1) * 3;
}

// Convert one float32 to float16 bit pattern.  Typical 3DGS inputs (linear
// scales, SH coefficients, DC/opacity) are well within the fp16 dynamic range.
// Denormals are flushed to zero for simplicity.
static std::uint16_t F32ToF16(float f) {
    std::uint32_t b;
    std::memcpy(&b, &f, sizeof(b));
    const std::uint32_t sign = (b >> 31) & 0x1u;
    const std::int32_t exp32 = static_cast<std::int32_t>((b >> 23) & 0xFFu);
    const std::uint32_t mant32 = b & 0x7FFFFFu;
    if (exp32 == 0xFF) {  // Inf or NaN
        return static_cast<std::uint16_t>((sign << 15) | 0x7C00u |
                                          (mant32 ? 0x200u : 0u));
    }
    const std::int32_t exp16 = exp32 - 127 + 15;
    if (exp16 >= 31) {
        return static_cast<std::uint16_t>((sign << 15) | 0x7C00u);
    }
    if (exp16 <= 0) {
        return static_cast<std::uint16_t>(sign << 15);
    }
    return static_cast<std::uint16_t>(
            (sign << 15) | (static_cast<std::uint32_t>(exp16) << 10) |
            (mant32 >> 13));
}

// Pack two fp32 values as fp16 into one uint32 (lo in bits 0-15, hi in 16-31).
// Matches the bit layout read by GLSL unpackHalf2x16().
static std::uint32_t PackHalf2(float lo, float hi) {
    return static_cast<std::uint32_t>(F32ToF16(lo)) |
           (static_cast<std::uint32_t>(F32ToF16(hi)) << 16);
}

// Encode a float in [-1, 1] as a biased uint8 (stored value =
// int8(round(f*127)) + 128). GPU decode: (float(byte) - 128.0) / 127.0.
static std::uint8_t EncodeSnorm8(float f) {
    const float clamped = std::max(-1.0f, std::min(1.0f, f));
    const auto q = static_cast<std::int8_t>(
            static_cast<int>(std::round(clamped * 127.0f)));
    return static_cast<std::uint8_t>(static_cast<int>(q) + 128);
}

// Pack quaternion components (w, x, y, z) as 4 biased snorm8 values into one
// uint32 (w in bits 0-7, x in 8-15, y in 16-23, z in 24-31).
static std::uint32_t PackSnorm8x4(float w, float x, float y, float z) {
    return static_cast<std::uint32_t>(EncodeSnorm8(w)) |
           (static_cast<std::uint32_t>(EncodeSnorm8(x)) << 8) |
           (static_cast<std::uint32_t>(EncodeSnorm8(y)) << 16) |
           (static_cast<std::uint32_t>(EncodeSnorm8(z)) << 24);
}

}  // namespace

PackedGaussianScene PackGaussianViewParams(
        const GaussianSplatPackedAttrs& attrs,
        const GaussianSplatRenderer::ViewRenderData& render_data,
        const GaussianSplatRenderer::RenderConfig& config) {
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

    // ---- Fill view-params UBO (288 bytes uploaded to GPU every frame) ----
    auto& vp = packed.view_params;
    std::memset(&vp, 0, sizeof(vp));

    const std::uint64_t entry_budget =
            static_cast<std::uint64_t>(n) * config.max_tiles_per_splat;
    const std::uint32_t entry_capacity = static_cast<std::uint32_t>(std::min(
            entry_budget,
            static_cast<std::uint64_t>(config.max_tile_entries_total)));

    // world_from_model: geometry model-to-world transform.
    // Gaussian splat positions are already in world space, so this is identity.
    // Note: render_data.model_matrix is the CAMERA's model matrix (camera-to-
    // world), which must NOT be used here.
    Eigen::Matrix4f model_f = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f view_f = render_data.view_matrix.matrix().cast<float>();
    Eigen::Matrix4f proj_f = render_data.projection_matrix.matrix();
    PackMat4(vp.world_from_model, model_f);
    PackMat4(vp.view_from_world, view_f);
    PackMat4(vp.clip_from_view, proj_f);

    vp.camera_position_and_near[0] = render_data.camera_position.x();
    vp.camera_position_and_near[1] = render_data.camera_position.y();
    vp.camera_position_and_near[2] = render_data.camera_position.z();
    vp.camera_position_and_near[3] = static_cast<float>(render_data.near_plane);

    vp.viewport_origin_and_size[0] =
            static_cast<float>(render_data.viewport_origin.x());
    vp.viewport_origin_and_size[1] =
            static_cast<float>(render_data.viewport_origin.y());
    vp.viewport_origin_and_size[2] = static_cast<float>(w);
    vp.viewport_origin_and_size[3] = static_cast<float>(h);

    const int effective_sh_degree =
            std::min(attrs.sh_degree, config.max_sh_degree);
    vp.scene[0] = n;
    vp.scene[1] = static_cast<std::uint32_t>(effective_sh_degree);
    // Antialias: enabled if requested by the material (per-scene) or the
    // renderer-level RenderConfig override.
    vp.scene[2] = (attrs.antialias || config.antialias) ? 1u : 0u;
    vp.scene[3] = render_data.screen_y_down ? 1u : 0u;

    vp.tiles[0] = static_cast<std::uint32_t>(tx);
    vp.tiles[1] = static_cast<std::uint32_t>(ty);
    vp.tiles[2] = static_cast<std::uint32_t>(tcx);
    vp.tiles[3] = static_cast<std::uint32_t>(tcy);

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

    vp.depth_range_and_flags[0] = static_cast<float>(render_data.near_plane);
    vp.depth_range_and_flags[1] = static_cast<float>(render_data.far_plane);
    vp.depth_range_and_flags[2] = 0.0f;  // reserved
    vp.depth_range_and_flags[3] = 0.0f;

    packed.splat_count = n;
    packed.tile_count = static_cast<std::uint32_t>(tcx * tcy);
    packed.pixel_count = static_cast<std::uint32_t>(w * h);
    packed.valid = true;
    return packed;
}

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
                                  GaussianSplatPackedAttrs& out) {
    out = GaussianSplatPackedAttrs{};
    out.sh_degree = desired_sh_degree;
    out.min_alpha = min_alpha;
    out.antialias = antialias;

    // Determine SH rest-coefficient counts and packed stride.
    // source_coeffs: stride in f_rest_ptr per source splat.
    // desired_coeffs: how many coefficients to pack per output splat.
    // sh_u32_per_splat: packed u32 count per splat (= 3 * degree * 2).
    int source_coeffs = 0;
    int desired_coeffs = 0;
    int sh_u32_per_splat = 0;
    if (desired_sh_degree > 0 && f_rest_ptr) {
        source_coeffs = GetGaussianSplatRestCoeffCount(source_sh_degree);
        desired_coeffs = GetGaussianSplatRestCoeffCount(desired_sh_degree);
        sh_u32_per_splat = 3 * desired_sh_degree * 2;
    }

    // Reserve upper bound; actual count may be lower after opacity filtering.
    out.positions.reserve(n);
    out.scales.reserve(2 * n);
    out.rotations.reserve(n);
    out.dc_opacity.reserve(2 * n);
    if (sh_u32_per_splat > 0) {
        out.sh_coefficients.reserve(static_cast<std::size_t>(sh_u32_per_splat) *
                                    n);
    }

    for (std::size_t i = 0; i < n; ++i) {
        // Filter: skip splats below the opacity threshold.
        const float opacity = opacity_ptr ? opacity_ptr[i] : 0.f;
        if (opacity < min_opacity_logit) continue;

        // Position: fp32 vec4 (w=0 padding for std430 alignment).
        out.positions.push_back({pts_ptr[i * 3 + 0], pts_ptr[i * 3 + 1],
                                 pts_ptr[i * 3 + 2], 0.f});

        // Linear scales: two fp16 pairs packed into uvec2 (8 B/splat).
        // scale_ptr holds linear values (exp already applied by the caller).
        if (scale_ptr) {
            out.scales.push_back(
                    PackHalf2(scale_ptr[i * 3 + 0], scale_ptr[i * 3 + 1]));
            out.scales.push_back(PackHalf2(scale_ptr[i * 3 + 2], 0.0f));
        } else {
            out.scales.push_back(PackHalf2(0.f, 0.f));
            out.scales.push_back(PackHalf2(0.f, 0.f));
        }

        // Rotation quaternion: snorm8-biased ×4 packed into one u32 (4
        // B/splat).
        if (rot_ptr) {
            out.rotations.push_back(
                    PackSnorm8x4(rot_ptr[i * 4 + 0], rot_ptr[i * 4 + 1],
                                 rot_ptr[i * 4 + 2], rot_ptr[i * 4 + 3]));
        } else {
            // Identity quaternion (w=1, x=y=z=0).
            out.rotations.push_back(PackSnorm8x4(1.f, 0.f, 0.f, 0.f));
        }

        // DC color + opacity: two fp16 pairs packed into uvec2 (8 B/splat).
        if (f_dc_ptr && opacity_ptr) {
            out.dc_opacity.push_back(
                    PackHalf2(f_dc_ptr[i * 3 + 0], f_dc_ptr[i * 3 + 1]));
            out.dc_opacity.push_back(
                    PackHalf2(f_dc_ptr[i * 3 + 2], opacity_ptr[i]));
        } else {
            out.dc_opacity.push_back(PackHalf2(0.f, 0.f));
            out.dc_opacity.push_back(PackHalf2(0.f, 0.f));
        }

        // SH rest coefficients: fp16 pairs packed into uvec2 array.
        // Packs exactly sh_u32_per_splat u32s (including zero padding) so the
        // buffer has uniform stride for GPU indexing.
        if (sh_u32_per_splat > 0 && f_rest_ptr) {
            const float* src = f_rest_ptr + i * source_coeffs;
            const int pairs = desired_coeffs / 2;
            for (int p = 0; p < pairs; ++p) {
                out.sh_coefficients.push_back(
                        PackHalf2(src[p * 2 + 0], src[p * 2 + 1]));
            }
            // Odd trailing coefficient (degree-1 has 9 coefficients).
            if (desired_coeffs & 1) {
                out.sh_coefficients.push_back(
                        PackHalf2(src[desired_coeffs - 1], 0.0f));
            }
            // Zero-pad remaining u32 slots to maintain uniform stride.
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
