// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Implementation of data packing helpers for the Gaussian splat compute
// renderer.

#include "open3d/visualization/rendering/filament/GaussianComputeDataPacking.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace open3d {
namespace visualization {
namespace rendering {

namespace {

// Pack an Eigen 4x4 (column-major) into a 16-float array in column-major order
// matching std140 mat4 layout.
void PackMat4(float* dst, const Eigen::Matrix4f& m) {
    std::memcpy(dst, m.data(), 16 * sizeof(float));
}

int CeilDiv(int value, int divisor) { return (value + divisor - 1) / divisor; }

// Convert one float32 to float16 bit pattern.  Typical 3DGS inputs (log-scales,
// SH coefficients, DC/opacity) are well within the fp16 dynamic range.
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
        const GaussianSplatSourceData& source,
        const GaussianComputeRenderer::ViewRenderData& render_data,
        const GaussianComputeRenderer::RenderConfig& config) {
    PackedGaussianScene packed;

    if (source.splat_count == 0 || render_data.viewport_size.x() <= 0 ||
        render_data.viewport_size.y() <= 0) {
        return packed;
    }

    const std::uint32_t n = source.splat_count;
    const int w = render_data.viewport_size.x();
    const int h = render_data.viewport_size.y();
    const int tx = config.tile_size.x();
    const int ty = config.tile_size.y();
    const int tcx = CeilDiv(w, tx);
    const int tcy = CeilDiv(h, ty);

    // ---- Fill view-params UBO (208 bytes uploaded to GPU every frame) ----
    auto& vp = packed.view_params;
    std::memset(&vp, 0, sizeof(vp));

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

    vp.scene[0] = n;
    vp.scene[1] = static_cast<std::uint32_t>(
            std::min(source.gaussian_splat_sh_degree, config.max_sh_degree));
    // Antialias: enabled if requested by the material (per-scene) or the
    // renderer-level RenderConfig override.
    vp.scene[2] =
            (source.gaussian_splat_antialias || config.antialias) ? 1u : 0u;
    vp.scene[3] = 0;

    vp.tiles[0] = static_cast<std::uint32_t>(tx);
    vp.tiles[1] = static_cast<std::uint32_t>(ty);
    vp.tiles[2] = static_cast<std::uint32_t>(tcx);
    vp.tiles[3] = static_cast<std::uint32_t>(tcy);

    vp.depth_range_and_flags[0] = static_cast<float>(render_data.near_plane);
    vp.depth_range_and_flags[1] = static_cast<float>(render_data.far_plane);
    vp.depth_range_and_flags[2] = config.stable_sort ? 1.0f : 0.0f;
    vp.depth_range_and_flags[3] = 0.0f;

    packed.splat_count = n;
    packed.tile_count = static_cast<std::uint32_t>(tcx * tcy);
    packed.pixel_count = static_cast<std::uint32_t>(w * h);
    packed.valid = true;
    return packed;
}

void PackGaussianSceneAttributes(
        const GaussianSplatSourceData& source,
        const GaussianComputeRenderer::RenderConfig& config,
        PackedGaussianScene& packed) {
    const std::uint32_t n = source.splat_count;

    // ---- positions: fp32 vec4[], 16 B/splat (layout unchanged) ----
    packed.positions.resize(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        packed.positions[i].x = source.positions[i * 3 + 0];
        packed.positions[i].y = source.positions[i * 3 + 1];
        packed.positions[i].z = source.positions[i * 3 + 2];
        packed.positions[i].w = 0.f;
    }

    // ---- log_scales: fp16 uvec2[], 8 B/splat ----
    // Two uint32 per splat: [0]=PackHalf2(sx,sy), [1]=PackHalf2(sz,0).
    // Shader reads: uvec2 p=log_scales[i]; (xy)=unpackHalf2x16(p.x);
    // z=unpackHalf2x16(p.y).x
    packed.log_scales.resize(2 * n);
    for (std::uint32_t i = 0; i < n; ++i) {
        packed.log_scales[2 * i + 0] = PackHalf2(source.log_scales[i * 3 + 0],
                                                 source.log_scales[i * 3 + 1]);
        packed.log_scales[2 * i + 1] =
                PackHalf2(source.log_scales[i * 3 + 2], 0.0f);
    }

    // ---- rotations: snorm8-biased uint[], 4 B/splat ----
    // Quaternion (w,x,y,z) in bytes 0/1/2/3; stored byte =
    // int8(round(q*127))+128. Shader decode: q = normalize((float[4](bytes) -
    // 128.0) / 127.0)
    packed.rotations.resize(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        packed.rotations[i] = PackSnorm8x4(
                source.rotations[i * 4 + 0], source.rotations[i * 4 + 1],
                source.rotations[i * 4 + 2], source.rotations[i * 4 + 3]);
    }

    // ---- dc_opacity: fp16 uvec2[], 8 B/splat ----
    // Two uint32 per splat: [0]=PackHalf2(r,g), [1]=PackHalf2(b,opacity).
    packed.dc_opacity.resize(2 * n);
    for (std::uint32_t i = 0; i < n; ++i) {
        packed.dc_opacity[2 * i + 0] = PackHalf2(source.dc_opacity[i * 4 + 0],
                                                 source.dc_opacity[i * 4 + 1]);
        packed.dc_opacity[2 * i + 1] = PackHalf2(source.dc_opacity[i * 4 + 2],
                                                 source.dc_opacity[i * 4 + 3]);
    }

    // ---- sh_coefficients: fp16 uvec2[], degree-dependent ----
    // Stride = 3 * sh_degree uvec2 per splat (i.e., 3*degree uint32 pairs):
    //   degree=0: 0 B/splat  (buffer empty; shader branch is never taken)
    //   degree=1: 24 B/splat (3 uvec2; 9 coeffs packed into 12 fp16 slots)
    //   degree=2: 48 B/splat (6 uvec2; 24 coeffs, no padding needed)
    // Each uvec2 stores 4 fp16 values matching the old fp32 vec4 index so
    // the shader's LoadShVec4(splat, j) indexes identically.
    const int sh_degree =
            std::min(source.gaussian_splat_sh_degree, config.max_sh_degree);
    if (sh_degree == 0 || source.sh_rest.empty()) {
        packed.sh_coefficients.clear();
    } else {
        // sh_u32_per_splat = 3 * degree * 2  (3 uvec2 * 2 uint32)
        const int sh_u32_per_splat = 3 * sh_degree * 2;
        const int sh_coeffs_per_splat = sh_degree * (sh_degree + 2) * 3;
        packed.sh_coefficients.assign(
                static_cast<std::size_t>(sh_u32_per_splat) * n, 0u);
        for (std::uint32_t i = 0; i < n; ++i) {
            const float* src = source.sh_rest.data() + i * sh_coeffs_per_splat;
            std::uint32_t* dst =
                    packed.sh_coefficients.data() + i * sh_u32_per_splat;
            // Pack consecutive pairs of fp32 SH coefficients as fp16.
            const int pairs = sh_coeffs_per_splat / 2;
            for (int p = 0; p < pairs; ++p) {
                dst[p] = PackHalf2(src[p * 2 + 0], src[p * 2 + 1]);
            }
            // Degree-1 has 9 coefficients (odd count); store the trailing one.
            if (sh_coeffs_per_splat & 1) {
                dst[pairs] = PackHalf2(src[sh_coeffs_per_splat - 1], 0.0f);
            }
        }
    }
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
