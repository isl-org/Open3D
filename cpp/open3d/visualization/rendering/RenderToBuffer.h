// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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
        const std::uint8_t* bytes = nullptr;
        std::size_t size = 0;
    };

    using BufferReadyCallback = std::function<void(const Buffer&)>;

    virtual ~RenderToBuffer() = default;

    // BufferReadyCallback does not need to free Buffer::bytes.
    // It should also not cache the pointer.
    virtual void Configure(const View* view,
                           Scene* scene,
                           int width,
                           int height,
                           BufferReadyCallback cb) = 0;
    virtual void SetDimensions(std::uint32_t width, std::uint32_t height) = 0;
    virtual View& GetView() = 0;

    virtual void Render() = 0;
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
