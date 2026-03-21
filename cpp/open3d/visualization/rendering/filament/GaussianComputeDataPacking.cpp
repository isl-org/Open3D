// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Implementation of data packing helpers and texture upload for the Gaussian
// splat compute renderer.

#include "open3d/visualization/rendering/filament/GaussianComputeDataPacking.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4068 4146 4293)
#endif
#include <filament/Engine.h>
#include <filament/Texture.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <algorithm>
#include <cmath>
#include <cstring>

#include "open3d/utility/Logging.h"
#include "open3d/visualization/rendering/filament/FilamentEngine.h"
#include "open3d/visualization/rendering/filament/FilamentResourceManager.h"

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

}  // namespace

PackedGaussianScene PackGaussianSceneInputs(
        const GaussianSplatSourceData& source,
        const GaussianComputeRenderer::ViewRenderData& render_data,
        const GaussianComputeRenderer::RenderConfig& config,
        const std::vector<GaussianComputeRenderer::PassDispatch>&
                dispatches) {
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

    // ---- Fill view params ----
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
            std::min(source.sh_degree, config.max_sh_degree));
    vp.scene[2] = 0;
    vp.scene[3] = 0;

    vp.tiles[0] = static_cast<std::uint32_t>(tx);
    vp.tiles[1] = static_cast<std::uint32_t>(ty);
    vp.tiles[2] = static_cast<std::uint32_t>(tcx);
    vp.tiles[3] = static_cast<std::uint32_t>(tcy);

    vp.depth_range_and_flags[0] = static_cast<float>(render_data.near_plane);
    vp.depth_range_and_flags[1] = static_cast<float>(render_data.far_plane);
    vp.depth_range_and_flags[2] = config.stable_sort ? 1.0f : 0.0f;
    vp.depth_range_and_flags[3] = 0.0f;

    // ---- Pack positions as vec4[] ----
    packed.positions.resize(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        packed.positions[i].x = source.positions[i * 3 + 0];
        packed.positions[i].y = source.positions[i * 3 + 1];
        packed.positions[i].z = source.positions[i * 3 + 2];
        packed.positions[i].w = 0.f;
    }

    // ---- Pack log_scales as vec4[] ----
    packed.log_scales.resize(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        packed.log_scales[i].x = source.log_scales[i * 3 + 0];
        packed.log_scales[i].y = source.log_scales[i * 3 + 1];
        packed.log_scales[i].z = source.log_scales[i * 3 + 2];
        packed.log_scales[i].w = 0.f;
    }

    // ---- Pack rotations as vec4[] ----
    packed.rotations.resize(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        packed.rotations[i].x = source.rotations[i * 4 + 0];
        packed.rotations[i].y = source.rotations[i * 4 + 1];
        packed.rotations[i].z = source.rotations[i * 4 + 2];
        packed.rotations[i].w = source.rotations[i * 4 + 3];
    }

    // ---- Pack dc_opacity as vec4[] ----
    packed.dc_opacity.resize(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        packed.dc_opacity[i].x = source.dc_opacity[i * 4 + 0];
        packed.dc_opacity[i].y = source.dc_opacity[i * 4 + 1];
        packed.dc_opacity[i].z = source.dc_opacity[i * 4 + 2];
        packed.dc_opacity[i].w = source.dc_opacity[i * 4 + 3];
    }

    // ---- Pack SH coefficients ----
    // The shader loads 6 vec4 per splat (kMaxShVec4PerSplat = 6).
    static constexpr std::uint32_t kMaxShVec4PerSplat = 6;
    const std::uint32_t sh_vec4_total = n * kMaxShVec4PerSplat;
    packed.sh_coefficients.assign(sh_vec4_total, Std430Vec4{});

    if (source.sh_degree >= 1 && !source.sh_rest.empty()) {
        const int sh_degree = std::min(source.sh_degree, 2);
        const int sh_coeffs_per_splat = sh_degree * (sh_degree + 2) * 3;
        for (std::uint32_t i = 0; i < n; ++i) {
            const float* src = source.sh_rest.data() + i * sh_coeffs_per_splat;
            float* dst = reinterpret_cast<float*>(
                    &packed.sh_coefficients[i * kMaxShVec4PerSplat]);
            const int floats_to_copy =
                    std::min(sh_coeffs_per_splat,
                             static_cast<int>(kMaxShVec4PerSplat * 4));
            std::memcpy(dst, src, floats_to_copy * sizeof(float));
        }
    }

    packed.splat_count = n;
    packed.tile_count = static_cast<std::uint32_t>(tcx * tcy);
    packed.pixel_count = static_cast<std::uint32_t>(w * h);
    packed.valid = true;
    return packed;
}

bool UploadOutputTextures(
        FilamentResourceManager& resource_mgr,
        const std::vector<Std430Vec4>& color_pixels,
        const std::vector<float>& depth_pixels,
        std::uint32_t width,
        std::uint32_t height,
        GaussianComputeRenderer::OutputTargets& targets) {
    if (width == 0 || height == 0) {
        return false;
    }

    auto& engine = EngineInstance::GetInstance();

    // Upload color (RGBA16F texture) from float4 data.
    {
        auto tex_weak = resource_mgr.GetTexture(targets.color);
        auto tex = tex_weak.lock();
        if (!tex) {
            utility::LogWarning(
                    "Gaussian compute output: invalid color texture handle.");
            return false;
        }
        // Convert float RGBA to half-float RGBA for the RGBA16F texture.
        const std::size_t pixel_count =
                static_cast<std::size_t>(width) * height;
        const std::size_t half_data_size = pixel_count * 4 * sizeof(uint16_t);
        auto* half_data = static_cast<uint16_t*>(std::malloc(half_data_size));
        if (!half_data) return false;

        for (std::size_t i = 0; i < pixel_count; ++i) {
            const Std430Vec4& src =
                    (i < color_pixels.size())
                            ? color_pixels[i]
                            : *reinterpret_cast<const Std430Vec4*>(
                                      "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0");
            // Use Eigen's half conversion or just bit-approximate.
            auto float_to_half = [](float f) -> uint16_t {
                // IEEE 754 float16 conversion.
                uint32_t bits;
                std::memcpy(&bits, &f, 4);
                uint32_t sign = (bits >> 16) & 0x8000;
                int32_t exponent =
                        ((bits >> 23) & 0xFF) - 127 + 15;
                uint32_t mantissa = bits & 0x007FFFFF;
                if (exponent <= 0) {
                    return static_cast<uint16_t>(sign);
                }
                if (exponent >= 31) {
                    return static_cast<uint16_t>(sign | 0x7C00);
                }
                return static_cast<uint16_t>(
                        sign | (exponent << 10) |
                        (mantissa >> 13));
            };
            half_data[i * 4 + 0] = float_to_half(src.x);
            half_data[i * 4 + 1] = float_to_half(src.y);
            half_data[i * 4 + 2] = float_to_half(src.z);
            half_data[i * 4 + 3] = float_to_half(src.w);
        }

        filament::Texture::PixelBufferDescriptor desc(
                half_data, half_data_size,
                filament::Texture::Format::RGBA,
                filament::Texture::Type::HALF,
                [](void* buf, size_t, void*) { std::free(buf); });
        tex->setImage(engine, 0, std::move(desc));
    }

    // Upload depth (DEPTH32F texture) - but Filament DEPTH32F textures only
    // support DEPTH_ATTACHMENT usage, not SAMPLEABLE+setImage upload.
    // Instead, we mark the output as valid and the depth buffer can be used
    // through the render target path.
    // For now, color-only output is functional.
    targets.has_valid_output = true;
    return true;
}

bool UploadOutputTexturesHalf(
        FilamentResourceManager& resource_mgr,
        const void* half_rgba_data,
        std::size_t half_data_size,
        std::uint32_t width,
        std::uint32_t height,
        GaussianComputeRenderer::OutputTargets& targets) {
    if (width == 0 || height == 0 || !half_rgba_data || half_data_size == 0) {
        return false;
    }

    auto& engine = EngineInstance::GetInstance();

    auto tex_weak = resource_mgr.GetTexture(targets.color);
    auto tex = tex_weak.lock();
    if (!tex) {
        utility::LogWarning(
                "Gaussian compute output: invalid color texture handle.");
        return false;
    }

    // Allocate a copy of the half data that Filament can own and free.
    auto* owned_data = static_cast<uint16_t*>(std::malloc(half_data_size));
    if (!owned_data) return false;
    std::memcpy(owned_data, half_rgba_data, half_data_size);

    filament::Texture::PixelBufferDescriptor desc(
            owned_data, half_data_size, filament::Texture::Format::RGBA,
            filament::Texture::Type::HALF,
            [](void* buf, size_t, void*) { std::free(buf); });
    tex->setImage(engine, 0, std::move(desc));

    targets.has_valid_output = true;
    return true;
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
