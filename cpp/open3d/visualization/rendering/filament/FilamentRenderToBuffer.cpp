// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
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

#include <algorithm>
#include <cstdint>
#include <vector>

#include "open3d/utility/Logging.h"
#include "open3d/visualization/rendering/filament/FilamentEngine.h"
#include "open3d/visualization/rendering/filament/FilamentRenderer.h"
#include "open3d/visualization/rendering/filament/FilamentResourceManager.h"
#include "open3d/visualization/rendering/filament/FilamentScene.h"
#include "open3d/visualization/rendering/filament/FilamentView.h"
#include "open3d/visualization/rendering/filament/GaussianSplatFrameScheduler.h"
#include "open3d/visualization/rendering/filament/GaussianSplatRenderer.h"

namespace open3d {
namespace visualization {
namespace rendering {

namespace {

/// Composite shader stores premultiplied RGB in \p gs_rgba; blend like ImGui
/// \c One / \c OneMinusSrcAlpha over an opaque Filament base.
void BlendPremultipliedSplatOverRgb8(
        uint8_t* base_rgb, int n_channels, const float* gs_rgba, int w, int h) {
    const int n = w * h;
    for (int i = 0; i < n; ++i) {
        const float fr = gs_rgba[i * 4 + 0];
        const float fg = gs_rgba[i * 4 + 1];
        const float fb = gs_rgba[i * 4 + 2];
        const float fa = gs_rgba[i * 4 + 3];
        const float br = static_cast<float>(base_rgb[i * n_channels + 0]) *
                         (1.f / 255.f);
        const float bg = static_cast<float>(base_rgb[i * n_channels + 1]) *
                         (1.f / 255.f);
        const float bb = static_cast<float>(base_rgb[i * n_channels + 2]) *
                         (1.f / 255.f);
        const float r = fr + br * (1.f - fa);
        const float g = fg + bg * (1.f - fa);
        const float b = fb + bb * (1.f - fa);
        base_rgb[i * n_channels + 0] = static_cast<uint8_t>(
                std::min(std::max(r, 0.f), 1.f) * 255.f + 0.5f);
        base_rgb[i * n_channels + 1] = static_cast<uint8_t>(
                std::min(std::max(g, 0.f), 1.f) * 255.f + 0.5f);
        base_rgb[i * n_channels + 2] = static_cast<uint8_t>(
                std::min(std::max(b, 0.f), 1.f) * 255.f + 0.5f);
        if (n_channels == 4) {
            base_rgb[i * n_channels + 3] = 255;
        }
    }
}

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

using PBDParams = std::tuple<FilamentRenderToBuffer*,
                             FilamentRenderToBuffer::BufferReadyCallback>;

void FilamentRenderToBuffer::ReadPixelsCallback(void*, size_t, void* user) {
    auto params = static_cast<PBDParams*>(user);
    FilamentRenderToBuffer* self;
    BufferReadyCallback callback;
    std::tie(self, callback) = *params;

    callback({self->width_, self->height_, self->n_channels_, self->buffer_,
              self->buffer_size_});

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

    const bool has_gaussian =
            gaussian_splat_renderer_ && scene_->HasGaussianSplatGeometry();
    const bool run_gs_pipeline = has_gaussian;

    if (run_gs_pipeline) {
        gaussian_splat_renderer_->RequestRedrawForView(*view_);
        if (depth_image_) {
            // Signal that a depth readback is needed so the composite pass
            // allocates and populates the merged_depth_u16_tex scratch texture.
            gaussian_splat_renderer_->RequestDepthReadbackForView(*view_, true);
        }
        gaussian_splat_renderer_->BeginFrame();
#if !defined(__APPLE__)
        // Drain Filament work before Gaussian compute dispatches (shared
        // GL/Vulkan queue on non-Apple backends).
        engine_.flushAndWait();
#endif
        GaussianSplatFrameScheduler::RunGeometry(
                *gaussian_splat_renderer_, *view_, *scene_);
    }

    if (renderer_->beginFrame(swapchain_)) {
        renderer_->render(view_->GetNativeView());

        if (run_gs_pipeline &&
            !GaussianSplatFrameScheduler::CompositeRunsAfterFilamentEndFrame()) {
            engine_.flushAndWait();
            GaussianSplatFrameScheduler::RunComposite(*gaussian_splat_renderer_,
                                                      *view_);
        }

        using namespace filament;
        using namespace backend;

        auto vp = view_->GetNativeView()->getViewport();

        renderer_->endFrame();

        if (run_gs_pipeline &&
            GaussianSplatFrameScheduler::CompositeRunsAfterFilamentEndFrame()) {
            GaussianSplatFrameScheduler::RunComposite(*gaussian_splat_renderer_,
                                                      *view_);
        }

        engine_.flushAndWait();

        auto* resource_mgr = &EngineInstance::GetResourceManager();

        RenderTargetHandle view_rt_h = view_->GetRenderTargetHandle();
        filament::RenderTarget* native_view_rt = nullptr;
        if (view_rt_h) {
            auto weak_vrt = resource_mgr->GetRenderTarget(view_rt_h);
            if (auto vrt = weak_vrt.lock()) {
                native_view_rt = vrt.get();
            }
        }

        RenderTargetHandle gs_rt =
                run_gs_pipeline
                        ? gaussian_splat_renderer_->GetColorReadbackRT(*view_)
                        : RenderTargetHandle();
        filament::RenderTarget* native_gs_rt = nullptr;
        if (gs_rt) {
            auto weak_rt = resource_mgr->GetRenderTarget(gs_rt);
            if (auto rt_sptr = weak_rt.lock()) {
                native_gs_rt = rt_sptr.get();
            }
        }

        if (!depth_image_ && run_gs_pipeline && native_view_rt) {
            // Issue both readPixels (base + GS overlay) together, then do one
            // more flushAndWait to collect both callbacks synchronously.
            //
            // Metal readPixels from a render target only supports RGBA+UBYTE,
            // not RGB+UBYTE (Metal has no native RGB texture format).  Always
            // read RGBA8 for the base and strip alpha when n_channels_==3.
            // On GL, RGBA also works fine — use one path for both backends.

            const size_t n_pixels = static_cast<size_t>(width_) * height_;

            // Scratch buffers for the two parallel readPixels callbacks.
            std::vector<uint8_t> base_rgba(n_pixels * 4);
            std::vector<float> gs_f32;

            PixelBufferDescriptor base_pd(
                    base_rgba.data(), base_rgba.size(), PixelDataFormat::RGBA,
                    PixelDataType::UBYTE, [](void*, size_t, void*) {}, nullptr);
            renderer_->readPixels(native_view_rt, vp.left, vp.bottom, vp.width,
                                  vp.height, std::move(base_pd));

            if (native_gs_rt) {
                gs_f32.resize(n_pixels * 4);
                PixelBufferDescriptor gs_pd(
                        gs_f32.data(), gs_f32.size() * sizeof(float),
                        PixelDataFormat::RGBA, PixelDataType::FLOAT,
                        [](void*, size_t, void*) {}, nullptr);
                renderer_->readPixels(native_gs_rt, vp.left, vp.bottom,
                                      vp.width, vp.height, std::move(gs_pd));
            }

            // One more flush ensures both callbacks complete before we proceed.
            engine_.flushAndWait();

            // Unpack RGBA8 base → output buffer (strip alpha for RGB).
            const uint8_t* src = base_rgba.data();
            uint8_t* dst = buffer_;
            const int nc = static_cast<int>(n_channels_);
            const int np = static_cast<int>(n_pixels);
            for (int i = 0; i < np; ++i) {
                dst[i * nc + 0] = src[i * 4 + 0];
                dst[i * nc + 1] = src[i * 4 + 1];
                dst[i * nc + 2] = src[i * 4 + 2];
                if (nc == 4) dst[i * nc + 3] = src[i * 4 + 3];
            }
            if (native_gs_rt && !gs_f32.empty()) {
                BlendPremultipliedSplatOverRgb8(buffer_, nc, gs_f32.data(),
                                                int(width_), int(height_));
            }

            // Deliver result now; the BeginFrame flushAndWait is a no-op since
            // all GPU work has already been collected above.
            if (callback_) {
                callback_({static_cast<std::size_t>(width_),
                           static_cast<std::size_t>(height_),
                           static_cast<std::size_t>(n_channels_), buffer_,
                           buffer_size_});
                callback_ = nullptr;
            }
            frame_done_ = true;
        } else if (depth_image_ && run_gs_pipeline &&
                   gaussian_splat_renderer_) {
            // GPU-merged depth path: the composite pass has already merged
            // GS and Filament depth into a normalised R16UI texture.
            // Read it back directly — no CPU merge required.
            std::vector<std::uint16_t> merged_u16;
            const bool got_merged =
                    gaussian_splat_renderer_->ReadMergedDepthToUint16Cpu(
                            *view_, merged_u16,
                            static_cast<std::uint32_t>(width_),
                            static_cast<std::uint32_t>(height_)) &&
                    merged_u16.size() == width_ * height_;
            if (got_merged) {
                // Convert normalised uint16 [0,65535] → linear float [0,far].
                const float far_z =
                        static_cast<float>(view_->GetCamera()->GetFar());
                float* dst = reinterpret_cast<float*>(buffer_);
                for (size_t i = 0; i < merged_u16.size(); ++i) {
                    dst[i] = (merged_u16[i] / 65535.f) * far_z;
                }
                if (callback_) {
                    callback_({static_cast<std::size_t>(width_),
                               static_cast<std::size_t>(height_), 1u, buffer_,
                               buffer_size_});
                    callback_ = nullptr;
                }
                frame_done_ = true;
            } else {
                // Try GS-only composite depth (R32F) when no scene depth
                // was available for merging.
                std::vector<float> gs_depth;
                const bool got_gs_depth =
                        gaussian_splat_renderer_->ReadCompositeDepthToFloatCpu(
                                *view_, gs_depth,
                                static_cast<std::uint32_t>(width_),
                                static_cast<std::uint32_t>(height_)) &&
                        gs_depth.size() == width_ * height_;
                if (got_gs_depth) {
                    float* dst = reinterpret_cast<float*>(buffer_);
                    std::copy(gs_depth.begin(), gs_depth.end(), dst);
                    if (callback_) {
                        callback_({static_cast<std::size_t>(width_),
                                   static_cast<std::size_t>(height_), 1u,
                                   buffer_, buffer_size_});
                        callback_ = nullptr;
                    }
                    frame_done_ = true;
                } else {
                    // Final fallback: Filament depth only via readPixels
                    // (backend unsupported or no GS depth available).
                    auto* user_param = new PBDParams(this, callback_);
                    PixelBufferDescriptor pd(
                            buffer_, buffer_size_,
                            PixelDataFormat::DEPTH_COMPONENT, PixelDataType::FLOAT,
                            ReadPixelsCallback, user_param);
                    renderer_->readPixels(vp.left, vp.bottom, vp.width,
                                          vp.height, std::move(pd));
                }
            }
        } else {
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
            auto* user_param = new PBDParams(this, callback_);
            PixelBufferDescriptor pd(buffer_, buffer_size_, format, type,
                                     ReadPixelsCallback, user_param);
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
