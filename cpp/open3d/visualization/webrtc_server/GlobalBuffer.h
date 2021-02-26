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
            std::lock_guard<std::mutex> lock(bgra_buffer_mutex);
            return bgra_buffer_.Clone();
        }
    }

    void Write(const core::Tensor& rgb_buffer) {
        {
            std::lock_guard<std::mutex> lock(bgra_buffer_mutex);
            rgb_buffer.AssertShape(
                    {bgra_buffer_.GetShape(0), bgra_buffer_.GetShape(1), 3});
            rgb_buffer.AssertDtype(bgra_buffer_.GetDtype());
            rgb_buffer.AssertDevice(bgra_buffer_.GetDevice());
            bgra_buffer_.Slice(2, 0, 1) = rgb_buffer.Slice(2, 2, 3);
            bgra_buffer_.Slice(2, 1, 2) = rgb_buffer.Slice(2, 1, 2);
            bgra_buffer_.Slice(2, 2, 3) = rgb_buffer.Slice(2, 0, 1);
        }
    }

private:
    GlobalBuffer() {
        t::geometry::Image im;
        t::io::ReadImage(
                "/home/yixing/repo/Open3D/cpp/open3d/visualization/"
                "webrtc_server/html/lena_color_640_480.jpg",
                im);
        bgra_buffer_ = core::Tensor::Zeros({im.GetRows(), im.GetCols(), 4},
                                           im.GetDtype());
        bgra_buffer_.Slice(2, 0, 1) = im.AsTensor().Slice(2, 2, 3);
        bgra_buffer_.Slice(2, 1, 2) = im.AsTensor().Slice(2, 1, 2);
        bgra_buffer_.Slice(2, 2, 3) = im.AsTensor().Slice(2, 0, 1);
    }

    virtual ~GlobalBuffer() {}

    core::Tensor bgra_buffer_;  // bgra
    std::mutex bgra_buffer_mutex;
};

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
