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

#include <cstring>

#include "open3d/utility/Logging.h"
#include "open3d/visualization/rendering/filament/FilamentEngine.h"
#include "open3d/visualization/rendering/filament/FilamentRenderer.h"
#include "open3d/visualization/rendering/filament/FilamentScene.h"
#include "open3d/visualization/rendering/filament/FilamentView.h"

namespace open3d {
namespace visualization {
namespace rendering {

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

    // Filament's readPixels returns data with Y=0 at bottom (OpenGL
    // convention), but most image formats and display systems expect Y=0 at
    // top. Flip the Y-axis to match the behavior of Open3DViewer
    // (VisualizerRender).
    if (self->buffer_ && self->width_ > 0 && self->height_ > 0) {
        const std::size_t bytes_per_pixel =
                self->depth_image_ ? sizeof(std::float_t)
                                   : self->n_channels_ * sizeof(std::uint8_t);
        const std::size_t bytes_per_line = self->width_ * bytes_per_pixel;
        const std::size_t total_bytes = bytes_per_line * self->height_;

        // Allocate temporary buffer for flipped data
        std::uint8_t* flipped_buffer =
                static_cast<std::uint8_t*>(malloc(total_bytes));
        if (flipped_buffer) {
            // Copy rows in reverse order
            for (std::size_t i = 0; i < self->height_; i++) {
                std::size_t src_row = self->height_ - i - 1;
                std::memcpy(flipped_buffer + bytes_per_line * i,
                            self->buffer_ + bytes_per_line * src_row,
                            bytes_per_line);
            }

            // Replace buffer with flipped data
            std::memcpy(self->buffer_, flipped_buffer, total_bytes);
            free(flipped_buffer);
        }
    }

    callback({self->width_, self->height_, self->n_channels_, self->buffer_,
              self->buffer_size_});

    // Unassign the callback, in case it captured ourself. Then we would never
    // get freed.
    self->callback_ = nullptr;

    self->frame_done_ = true;
    delete params;
}

void FilamentRenderToBuffer::Render() {
    frame_done_ = false;
    scene_->HideRefractedMaterials();
    if (renderer_->beginFrame(swapchain_)) {
        renderer_->render(view_->GetNativeView());

        using namespace filament;
        using namespace backend;

        auto format = (n_channels_ == 3 ? PixelDataFormat::RGB
                                        : PixelDataFormat::RGBA);
        auto type = PixelDataType::UBYTE;
        if (depth_image_) {
            format = PixelDataFormat::DEPTH_COMPONENT;
            type = PixelDataType::FLOAT;
        }
        auto user_param = new PBDParams(this, callback_);
        PixelBufferDescriptor pd(buffer_, buffer_size_, format, type,
                                 ReadPixelsCallback, user_param);
        auto vp = view_->GetNativeView()->getViewport();

        renderer_->readPixels(vp.left, vp.bottom, vp.width, vp.height,
                              std::move(pd));

        renderer_->endFrame();
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
