// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 www.open3d.org
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
#include "open3d/utility/Console.h"
#include "open3d/visualization/rendering/RenderToBuffer.h"
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
            view, scene, vp[2], vp[3],
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
                render = nullptr;
            });
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
