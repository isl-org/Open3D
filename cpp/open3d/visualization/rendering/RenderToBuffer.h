// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <cstdint>
#include <functional>
#include <memory>

namespace open3d {
namespace visualization {
namespace rendering {

class Scene;
class View;

class RenderToBuffer {
public:
    struct Buffer {
        std::size_t width = 0;
        std::size_t height = 0;
        std::size_t n_channels = 0;
        const std::uint8_t* bytes = nullptr;
        std::size_t size = 0;
    };

    using BufferReadyCallback = std::function<void(const Buffer&)>;

    virtual ~RenderToBuffer() = default;

    // Sets a callback that will be called after rendering is finished
    // and after the BufferReadyCallback from Configure() is finished.
    // This callback can be used to deallocate the object. In particular,
    //   SetCleanupCallback([](RenderToBuffer *self) { delete self; });
    // is valid.
    void SetCleanupCallback(std::function<void(RenderToBuffer*)> cb) {
        cleanup_callback_ = cb;
    }

    // BufferReadyCallback does not need to free Buffer::bytes.
    // It should also not cache the pointer.
    virtual void Configure(const View* view,
                           Scene* scene,
                           int width,
                           int height,
                           int n_channels,
                           bool depth_image,
                           BufferReadyCallback cb) = 0;
    virtual void SetDimensions(std::uint32_t width, std::uint32_t height) = 0;
    virtual View& GetView() = 0;

    virtual void Render() = 0;

protected:
    std::function<void(RenderToBuffer*)> cleanup_callback_;
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
