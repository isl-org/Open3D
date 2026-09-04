// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/rendering/filament/FilamentRenderToBuffer.h"

// 4068: Filament has some clang-specific vectorizing pragma's that MSVC flags
// 4146: PixelBufferDescriptor assert unsigned is positive before subtracting
//       but MSVC can't figure that out.
// 4293: Filament's utils/algorithm.h utils::details::clz() does strange
//       things with MSVC. Somehow sizeof(unsigned int) > 4, but its size is
//       32 so that x >> 32 gives a warning. (Or maybe the compiler can't
//       determine the if statement does not run.)
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4068 4146 4293)
#endif  // _MSC_VER

#include <filament/Engine.h>
#include <filament/RenderableManager.h>
#include <filament/Renderer.h>
#include <filament/Scene.h>
#include <filament/SwapChain.h>
#include <filament/Texture.h>
#include <filament/View.h>
#include <filament/Viewport.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif  // _MSC_VER

#include <math/half.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "open3d/utility/Logging.h"
#include "open3d/visualization/rendering/filament/FilamentEngine.h"
#include "open3d/visualization/rendering/filament/FilamentRenderer.h"
#include "open3d/visualization/rendering/filament/FilamentResourceManager.h"
#include "open3d/visualization/rendering/filament/FilamentScene.h"
#include "open3d/visualization/rendering/filament/FilamentView.h"
#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatRenderer.h"

namespace open3d {
namespace visualization {
namespace rendering {

namespace {

float HalfBitsToFloat(std::uint16_t bits) {
    return static_cast<float>(filament::math::makeHalf(bits));
}

std::uint8_t ToUnorm8(float value) {
    return static_cast<std::uint8_t>(std::clamp(value, 0.0f, 1.0f) * 255.0f +
                                     0.5f);
}

#if defined(__APPLE__)
/// Composite shader stores premultiplied RGB in \p gs_rgba_bits; blend like
/// ImGui One / OneMinusSrcAlpha over an opaque Filament base.
void BlendPremultipliedSplatOverRgb8(uint8_t* base_rgb,
                                     int n_channels,
                                     const std::uint16_t* gs_rgba_bits,
                                     int n_pixels) {
    for (int i = 0; i < n_pixels; ++i) {
        uint8_t* base = base_rgb + i * n_channels;
        const std::uint16_t* splat = gs_rgba_bits + i * 4;
        const float splat_alpha = HalfBitsToFloat(splat[3]);
        for (int channel = 0; channel < 3; ++channel) {
            base[channel] =
                    ToUnorm8(HalfBitsToFloat(splat[channel]) +
                             base[channel] / 255.0f * (1.0f - splat_alpha));
        }
        if (n_channels == 4) base[3] = 255;
    }
}
#endif

}  // namespace

FilamentRenderToBuffer::FilamentRenderToBuffer(filament::Engine& engine)
    : engine_(engine) {
    renderer_ = engine_.createRenderer();
}

FilamentRenderToBuffer::~FilamentRenderToBuffer() {
    if (view_) delete view_;

    engine_.destroy(swapchain_);
    engine_.destroy(renderer_);

    if (buffer_) {
        free(buffer_);
        buffer_ = nullptr;

        buffer_size_ = 0;
    }
#if defined(__APPLE__)
    if (rgba_readback_buffer_) {
        free(rgba_readback_buffer_);
        rgba_readback_buffer_ = nullptr;
        rgba_readback_buffer_size_ = 0;
    }
#endif
}

void FilamentRenderToBuffer::Configure(const View* view,
                                       Scene* scene,
                                       int width,
                                       int height,
                                       int n_channels,
                                       bool depth_image,
                                       BufferReadyCallback cb) {
    if (!scene) {
        utility::LogDebug(
                "No Scene object was provided for rendering into buffer");
        cb({0, 0, 0, nullptr, 0});
        return;
    }

    if (pending_) {
        utility::LogWarning(
                "Render to buffer can process only one request at time");
        cb({0, 0, 0, nullptr, 0});
        return;
    }

    if (!depth_image && (n_channels != 3 && n_channels != 4)) {
        utility::LogWarning(
                "Render to buffer must have either 3 or 4 channels");
        cb({0, 0, 0, nullptr, 0});
        return;
    }

    if (depth_image) {
        n_channels_ = 1;
    } else {
        n_channels_ = n_channels;
    }
    depth_image_ = depth_image;
    pending_ = true;
    callback_ = cb;

    // Create a proper copy of the View with scen attached
    CopySettings(view);
    auto* downcast_scene = static_cast<FilamentScene*>(scene);
    if (downcast_scene) {
        view_->SetScene(*downcast_scene);
        scene_ = downcast_scene;
    }
    SetDimensions(width, height);
}

void FilamentRenderToBuffer::SetDimensions(const std::uint32_t width,
                                           const std::uint32_t height) {
    if (swapchain_) {
        engine_.destroy(swapchain_);
    }

    swapchain_ = engine_.createSwapChain(width, height,
                                         filament::SwapChain::CONFIG_READABLE);
    view_->SetViewport(0, 0, width, height);

    width_ = width;
    height_ = height;

    // Allocate cached Filament color/depth attachments for Gaussian splat
    // zero-copy and for readPixels of the Filament base pass.
    if (scene_ && scene_->HasGaussianSplatGeometry()) {
        view_->EnableViewCaching(true);
    }

    if (depth_image_) {
        buffer_size_ = width * height * sizeof(std::float_t);
    } else {
        buffer_size_ = width * height * n_channels_ * sizeof(std::uint8_t);
    }
    if (buffer_) {
        buffer_ = static_cast<std::uint8_t*>(realloc(buffer_, buffer_size_));
    } else {
        buffer_ = static_cast<std::uint8_t*>(malloc(buffer_size_));
    }
}

void FilamentRenderToBuffer::CopySettings(const View* view) {
    view_ = new FilamentView(engine_, EngineInstance::GetResourceManager());
    auto* downcast = static_cast<const FilamentView*>(view);
    if (downcast) {
        view_->CopySettingsFrom(*downcast);
    }
    if (depth_image_) {
        // Disable post-processing when rendering to depth image. It's uncessary
        // overhead and the depth buffer is discarded when post-processing is
        // enabled so the returned image is all 0s.
        view_->ConfigureForColorPicking();
        // Set shadowing to true as there is a pixel coordinate scaling
        // issue on Apple Retina displays that results in quarter size depth
        // images if shadowing is disabled.
        view_->SetShadowing(true, View::ShadowType::kPCF);
    }
}

View& FilamentRenderToBuffer::GetView() { return *view_; }

void FilamentRenderToBuffer::DeliverFrame(bool ok) {
    if (callback_) {
        callback_(
                ok ? Buffer{width_, height_, n_channels_, buffer_, buffer_size_}
                   : Buffer{0, 0, 0, nullptr, 0});
        callback_ = nullptr;
    }
    frame_done_ = true;
}

struct PBDParams {
    FilamentRenderToBuffer* self;
    FilamentRenderToBuffer::BufferReadyCallback callback;
    bool strip_rgba = false;
};

void FilamentRenderToBuffer::ReadPixelsCallback(void*, size_t, void* user) {
    auto params = static_cast<PBDParams*>(user);
    auto* self = params->self;

#if defined(__APPLE__)
    if (params->strip_rgba) {
        const std::size_t n_pixels = self->width_ * self->height_;
        for (std::size_t i = 0; i < n_pixels; ++i) {
            self->buffer_[i * 3 + 0] = self->rgba_readback_buffer_[i * 4 + 0];
            self->buffer_[i * 3 + 1] = self->rgba_readback_buffer_[i * 4 + 1];
            self->buffer_[i * 3 + 2] = self->rgba_readback_buffer_[i * 4 + 2];
        }
    }
#endif

    params->callback({self->width_, self->height_, self->n_channels_,
                      self->buffer_, self->buffer_size_});

    // Unassign the callback, in case it captured ourself. Then we would never
    // get freed.
    self->callback_ = nullptr;

    self->frame_done_ = true;
    delete params;
}

// Ordering mirrors FilamentRenderer::{BeginFrame,Draw,EndFrame}.
// Stage 1 (Geometry) runs before Filament's beginFrame.
// Stage 2 (Composite) runs after render() on non-Apple, after endFrame() on
// Apple.
void FilamentRenderToBuffer::Render() {
    frame_done_ = false;
    scene_->HideRefractedMaterials();

    const bool run_gs_pipeline =
            gaussian_splat_renderer_ && scene_->HasGaussianSplatGeometry();

    if (run_gs_pipeline) {
        gaussian_splat_renderer_->RequestRedrawForView(*view_);
        gaussian_splat_renderer_->BeginFrame();
#if !defined(__APPLE__)
        // Drain Filament work before Gaussian compute dispatches (shared
        // GL/Vulkan queue on non-Apple backends).
        engine_.flushAndWait();
#endif
        gaussian_splat_renderer_->RenderGeometryStage(*view_, *scene_);
        if (depth_image_) {
            // Signal that a depth readback is needed so the composite pass
            // allocates and populates the merged_depth_u16_tex scratch texture.
            // Must be after RenderGeometryStage which creates the OutputTargets
            // entry for this view via PrepareOutputTargets().
            gaussian_splat_renderer_->RequestDepthReadbackForView(*view_, true);
        }
    }

    if (renderer_->beginFrame(swapchain_)) {
        renderer_->render(view_->GetNativeView());

        using namespace filament;
        using namespace backend;

        auto vp = view_->GetNativeView()->getViewport();

        auto* resource_mgr = &EngineInstance::GetResourceManager();
        RenderTargetHandle view_rt_h = view_->GetRenderTargetHandle();
        filament::RenderTarget* native_view_rt = nullptr;
        if (view_rt_h) {
            auto weak_vrt = resource_mgr->GetRenderTarget(view_rt_h);
            if (auto vrt = weak_vrt.lock()) {
                native_view_rt = vrt.get();
            }
        }

        const size_t n_pixels = static_cast<size_t>(width_) * height_;
        const size_t n_gs_elems = n_pixels * 4;
        const int nc = static_cast<int>(n_channels_);

#if !defined(__APPLE__)
        renderer_->endFrame();
        engine_.flushAndWait();

        if (run_gs_pipeline) {
            gaussian_splat_renderer_->RenderCompositeStage(*view_);
        }

        // The composite pass blended splats in place into the shared RGBA16F
        // image, so that image alone is the finished colour frame.
        if (run_gs_pipeline && native_view_rt && !depth_image_) {
            std::vector<std::uint16_t> gs_rgba_bits;
            const bool got_gs_rgba =
                    gaussian_splat_renderer_->ReadColorToRGBA16FCpu(
                            *view_, gs_rgba_bits) &&
                    gs_rgba_bits.size() == n_gs_elems;

            engine_.flushAndWait();

            if (!got_gs_rgba) {
                utility::LogWarning(
                        "FilamentRenderToBuffer: Vulkan direct readback "
                        "failed; returning empty frame.");
                DeliverFrame(false);
            } else {
                for (size_t i = 0; i < n_pixels; ++i) {
                    for (int channel = 0; channel < std::min(nc, 4);
                         ++channel) {
                        buffer_[i * nc + channel] = ToUnorm8(
                                HalfBitsToFloat(gs_rgba_bits[i * 4 + channel]));
                    }
                }
                DeliverFrame();
            }
        }
#else
        renderer_->endFrame();
        if (run_gs_pipeline) {
            gaussian_splat_renderer_->RenderCompositeStage(*view_);
        }
        engine_.flushAndWait();

        // Metal keeps splats in a separate transparent overlay, so the base
        // scene and the overlay are read back and blended on the CPU.
        if (run_gs_pipeline && native_view_rt && !depth_image_) {
            std::vector<uint8_t> base_rgba(n_gs_elems, 0);
            PixelBufferDescriptor base_pd(
                    base_rgba.data(), base_rgba.size(), PixelDataFormat::RGBA,
                    PixelDataType::UBYTE, [](void*, size_t, void*) {}, nullptr);
            renderer_->readPixels(native_view_rt, vp.left, vp.bottom, vp.width,
                                  vp.height, std::move(base_pd));

            std::vector<std::uint16_t> gs_rgba_bits;
            bool got_gs_rgba = gaussian_splat_renderer_->ReadColorToRGBA16FCpu(
                                       *view_, gs_rgba_bits) &&
                               gs_rgba_bits.size() == n_gs_elems;
            RenderTargetHandle gs_rt =
                    gaussian_splat_renderer_->GetColorReadbackRT(*view_);
            if (!got_gs_rgba && gs_rt) {
                if (auto rt = resource_mgr->GetRenderTarget(gs_rt).lock()) {
                    gs_rgba_bits.assign(n_gs_elems, 0);
                    PixelBufferDescriptor gs_pd(
                            gs_rgba_bits.data(),
                            gs_rgba_bits.size() * sizeof(std::uint16_t),
                            PixelDataFormat::RGBA, PixelDataType::HALF,
                            [](void*, size_t, void*) {}, nullptr);
                    renderer_->readPixels(rt.get(), vp.left, vp.bottom,
                                          vp.width, vp.height,
                                          std::move(gs_pd));
                    got_gs_rgba = true;
                }
            }

            engine_.flushAndWait();

            for (size_t i = 0; i < n_pixels; ++i) {
                for (int channel = 0; channel < nc; ++channel) {
                    buffer_[i * nc + channel] = base_rgba[i * 4 + channel];
                }
            }
            if (got_gs_rgba) {
                BlendPremultipliedSplatOverRgb8(buffer_, nc,
                                                gs_rgba_bits.data(),
                                                static_cast<int>(n_pixels));
            }
            DeliverFrame();
        }
#endif

        if (!frame_done_ && depth_image_ && run_gs_pipeline) {
            // The composite pass already merged GS and Filament depth into a
            // normalised R16UI texture, so no CPU merge is required.
            // Renderer::RenderToDepthImage applies the final user-facing
            // conversion for z_in_view_space/normalized modes.
            float* dst = reinterpret_cast<float*>(buffer_);
            std::vector<std::uint16_t> merged_u16;
            std::vector<float> gs_depth;
            if (gaussian_splat_renderer_->ReadMergedDepthToUint16Cpu(
                        *view_, merged_u16, static_cast<std::uint32_t>(width_),
                        static_cast<std::uint32_t>(height_)) &&
                merged_u16.size() == n_pixels) {
                for (size_t i = 0; i < n_pixels; ++i) {
                    dst[i] = merged_u16[i] / 65535.f;
                }
                DeliverFrame();
            } else if (gaussian_splat_renderer_->ReadCompositeDepthToFloatCpu(
                               *view_, gs_depth,
                               static_cast<std::uint32_t>(width_),
                               static_cast<std::uint32_t>(height_)) &&
                       gs_depth.size() == n_pixels) {
                // GS-only depth when no scene depth was available for merging.
                std::copy(gs_depth.begin(), gs_depth.end(), dst);
                DeliverFrame();
            } else {
                // Final fallback: Filament depth only via readPixels.
                auto* user_param = new PBDParams{this, callback_};
                PixelBufferDescriptor pd(
                        buffer_, buffer_size_, PixelDataFormat::DEPTH_COMPONENT,
                        PixelDataType::FLOAT, ReadPixelsCallback, user_param);
                renderer_->readPixels(vp.left, vp.bottom, vp.width, vp.height,
                                      std::move(pd));
            }
        } else if (!frame_done_) {
            if (!depth_image_ && run_gs_pipeline && !native_view_rt) {
                utility::LogWarning(
                        "Gaussian splat offscreen: FilamentView has no render "
                        "target; expected EnableViewCaching. Reading the "
                        "swapchain — splat composite may be missing.");
            }
            auto format = (n_channels_ == 3 ? PixelDataFormat::RGB
                                            : PixelDataFormat::RGBA);
            auto type = PixelDataType::UBYTE;
            if (depth_image_) {
                format = PixelDataFormat::DEPTH_COMPONENT;
                type = PixelDataType::FLOAT;
            }
            auto* user_param = new PBDParams{this, callback_};
            void* readback_buffer = buffer_;
            size_t readback_size = buffer_size_;
#if defined(__APPLE__)
            if (!depth_image_ && n_channels_ == 3) {
                format = PixelDataFormat::RGBA;
                readback_size = width_ * height_ * 4;
                if (rgba_readback_buffer_size_ != readback_size) {
                    rgba_readback_buffer_ = static_cast<uint8_t*>(
                            realloc(rgba_readback_buffer_, readback_size));
                    rgba_readback_buffer_size_ = readback_size;
                }
                readback_buffer = rgba_readback_buffer_;
                user_param->strip_rgba = true;
            }
#endif
            PixelBufferDescriptor pd(readback_buffer, readback_size, format,
                                     type, ReadPixelsCallback, user_param);
            renderer_->readPixels(vp.left, vp.bottom, vp.width, vp.height,
                                  std::move(pd));
        }
    }
    scene_->HideRefractedMaterials(false);

    pending_ = false;
}

void FilamentRenderToBuffer::RenderTick() {
    if (renderer_->beginFrame(swapchain_)) {
        renderer_->endFrame();
    }
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
