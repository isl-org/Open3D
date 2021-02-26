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
            std::lock_guard<std::mutex> lock(is_new_frame_mutex);
            is_new_frame = false;
        }
        return rgb_buffer_;
    }

    void Write(const core::Tensor& rgb_buffer) {
        rgb_buffer.AssertShape(
                {rgb_buffer_.GetShape(0), rgb_buffer_.GetShape(1), 3});
        rgb_buffer.AssertDtype(rgb_buffer_.GetDtype());
        rgb_buffer.AssertDevice(rgb_buffer_.GetDevice());
        rgb_buffer_.AsRvalue() = rgb_buffer;
        {
            std::lock_guard<std::mutex> lock(is_new_frame_mutex);
            is_new_frame = true;
        }
    }

    // TODO: use proper "producer-consumer" model with signaling.
    // Currently we need a thread continuously pulling IsNewFrame() to read.
    bool IsNewFrame() { return is_new_frame; }

private:
    GlobalBuffer() {
        core::Tensor two_fifty_five =
                core::Tensor::Ones({}, core::Dtype::UInt8) * 255;
        rgb_buffer_ = core::Tensor::Zeros({480, 640, 3}, core::Dtype::UInt8);
        rgb_buffer_.Slice(0, 0, 160, 1).Slice(2, 0, 1, 1) = two_fifty_five;
        rgb_buffer_.Slice(0, 160, 320, 1).Slice(2, 1, 2, 1) = two_fifty_five;
        rgb_buffer_.Slice(0, 320, 480, 1).Slice(2, 2, 3, 1) = two_fifty_five;
    }

    virtual ~GlobalBuffer() {}

    core::Tensor rgb_buffer_;
    std::mutex is_new_frame_mutex;
    bool is_new_frame = false;
};

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
