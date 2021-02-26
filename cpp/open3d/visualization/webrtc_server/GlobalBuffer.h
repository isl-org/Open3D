// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2021 www.open3d.org
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
#include <mutex>

#include "open3d/core/Tensor.h"
#include "open3d/t/io/ImageIO.h"

namespace open3d {
namespace visualization {
namespace webrtc_server {

class GlobalBuffer {
public:
    static GlobalBuffer& GetInstance() {
        static GlobalBuffer instance;
        return instance;
    }

    core::Tensor Read() {
        {
            std::lock_guard<std::mutex> lock(rgb_buffer_mutex);
            return rgb_buffer_.Clone();
        }
    }

    void Write(const core::Tensor& rgb_buffer) {
        {
            std::lock_guard<std::mutex> lock(rgb_buffer_mutex);
            rgb_buffer.AssertShape(
                    {rgb_buffer_.GetShape(0), rgb_buffer_.GetShape(1), 3});
            rgb_buffer.AssertDtype(rgb_buffer_.GetDtype());
            rgb_buffer.AssertDevice(rgb_buffer_.GetDevice());
            rgb_buffer_.AsRvalue() = rgb_buffer;
        }
    }

private:
    GlobalBuffer() {
        t::geometry::Image im;
        t::io::ReadImage(
                "/home/yixing/repo/Open3D/cpp/open3d/visualization/"
                "webrtc_server/html/lena_color_640_480.jpg",
                im);
        rgb_buffer_ = im.AsTensor().Clone();
    }

    virtual ~GlobalBuffer() {}

    core::Tensor rgb_buffer_;  // bgra
    std::mutex rgb_buffer_mutex;
};

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
