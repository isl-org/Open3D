// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "open3d/visualization/rendering/Renderer.h"

#include "open3d/geometry/Image.h"
#include "open3d/utility/Logging.h"
#include "open3d/visualization/rendering/Material.h"
#include "open3d/visualization/rendering/RenderToBuffer.h"
#include "open3d/visualization/rendering/Scene.h"
#include "open3d/visualization/rendering/View.h"

namespace open3d {
namespace visualization {
namespace rendering {

static const ResourceLoadRequest::ErrorCallback kDefaultErrorHandler =
        [](const ResourceLoadRequest& request,
           const uint8_t code,
           const std::string& details) {
            if (!request.path_.empty()) {
                utility::LogWarning(
                        "Resource request for path {} failed:\n* Code: {}\n* "
                        "Error: {}",
                        request.path_.data(), code, details.data());
            } else if (request.data_size_ > 0) {
                utility::LogWarning(
                        "Resource request failed:\n* Code: {}\n * Error: {}",
                        code, details.data());
            } else {
                utility::LogWarning(
                        "Resource request failed: Malformed request");
            }
        };

ResourceLoadRequest::ResourceLoadRequest(const void* data, size_t data_size)
    : data_(data),
      data_size_(data_size),
      path_(""),
      error_callback_(kDefaultErrorHandler) {}

ResourceLoadRequest::ResourceLoadRequest(const char* path)
    : data_(nullptr),
      data_size_(0u),
      path_(path),
      error_callback_(kDefaultErrorHandler) {}

ResourceLoadRequest::ResourceLoadRequest(const void* data,
                                         size_t data_size,
                                         ErrorCallback error_callback)
    : data_(data),
      data_size_(data_size),
      path_(""),
      error_callback_(std::move(error_callback)) {}

ResourceLoadRequest::ResourceLoadRequest(const char* path,
                                         ErrorCallback error_callback)
    : data_(nullptr),
      data_size_(0u),
      path_(path),
      error_callback_(std::move(error_callback)) {}

void Renderer::RenderToImage(
        View* view,
        Scene* scene,
        std::function<void(std::shared_ptr<geometry::Image>)> cb) {
    auto vp = view->GetViewport();
    auto render = CreateBufferRenderer();
    render->Configure(
            view, scene, vp[2], vp[3], 3, false,
            // the shared_ptr (render) is const unless the lambda
            // is made mutable
            [render, cb](const RenderToBuffer::Buffer& buffer) mutable {
                auto image = std::make_shared<geometry::Image>();
                image->width_ = int(buffer.width);
                image->height_ = int(buffer.height);
                image->num_of_channels_ = 3;
                image->bytes_per_channel_ = 1;
                image->data_ = std::vector<uint8_t>(buffer.bytes,
                                                    buffer.bytes + buffer.size);
                cb(image);
                // This does not actually cause the object to be destroyed--
                // the lambda still has a copy--but it does ensure that the
                // object lives long enough for the callback to get executed.
                // The object will be freed when the callback is unassigned.
                render = nullptr;
            });
}

void Renderer::RenderToDepthImage(
        View* view,
        Scene* scene,
        std::function<void(std::shared_ptr<geometry::Image>)> cb,
        bool z_in_view_space /* = false */) {
    auto vp = view->GetViewport();
    auto render = CreateBufferRenderer();
    double z_near = view->GetCamera()->GetNear();
    render->Configure(
            view, scene, vp[2], vp[3], 1, true,
            // the shared_ptr (render) is const unless the lambda
            // is made mutable
            [render, cb, z_in_view_space,
             z_near](const RenderToBuffer::Buffer& buffer) mutable {
                auto image = std::make_shared<geometry::Image>();
                image->width_ = int(buffer.width);
                image->height_ = int(buffer.height);
                image->num_of_channels_ = 1;
                image->bytes_per_channel_ = 4;
                image->data_.resize(image->width_ * image->height_ *
                                    image->num_of_channels_ *
                                    image->bytes_per_channel_);
                memcpy(image->data_.data(), buffer.bytes, buffer.size);
                // Filament's shaders calculate depth ranging from 1
                // (near) to 0 (far), which is reversed from what is
                // normally done, so convert here so that users do not
                // need to rediscover Filament internals. (And so we
                // can change it back if Filament decides to swap how
                // they do it again.)
                float* pixels = (float*)image->data_.data();
                int n_pixels = image->width_ * image->height_;
                if (z_in_view_space) {
                    for (int i = 0; i < n_pixels; ++i) {
                        if (pixels[i] == 0.f) {
                            pixels[i] = std::numeric_limits<float>::infinity();
                        } else {
                            pixels[i] = z_near / pixels[i];
                        }
                    }
                } else {
                    for (int i = 0; i < n_pixels; ++i) {
                        pixels[i] = 1.0f - pixels[i];
                    }
                }

                cb(image);
                // This does not actually cause the object to be
                // destroyed-- the lambda still has a copy--but it
                // does ensure that the object lives long enough for
                // the callback to get executed. The object will be
                // freed when the callback is unassigned.
                render = nullptr;
            });
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
