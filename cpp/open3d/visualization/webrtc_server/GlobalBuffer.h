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
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "open3d/core/Tensor.h"
#include "open3d/t/io/ImageIO.h"

namespace open3d {
namespace visualization {
namespace webrtc_server {

class GlobalBuffer {
    static const int s_max_initial_frames_ = 10;

public:
    static GlobalBuffer& GetInstance() {
        static GlobalBuffer instance;
        return instance;
    }

    std::shared_ptr<core::Tensor> Read() {
        // At the initialization state, do not wait for the conditional
        // variables for the initial frames. This flushes the intial the WebRTC
        // stream for the client. The initialization state ends when the
        // s_max_initial_frames_ is reached or when the frame_ready_ is true.
        // Essentially, we are doing the "sleep and busy waiting" syncronization
        // at the initilization stage and then switch to conditional variables.
        //
        // If the initial frame rate is too high, WebRTC will potentailly drop
        // actual frames after the initilization phase is done. This leads to a
        // behavior of high-latency from the client's perspective when they
        // first see the actual rendering.
        //
        // TODO: (important) do not use the test pattern for the initial frames.
        // Instead, pro-actively get the rendered image from the renderer.
        if (num_initial_frames_ < s_max_initial_frames_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            std::unique_lock<std::mutex> ul(frame_mutex_);
            bool ready = frame_ready_;
            ul.unlock();
            if (ready) {
                num_initial_frames_ = s_max_initial_frames_;
            } else {
                num_initial_frames_++;
            }
            return frame_;
        }

        std::unique_lock<std::mutex> ul(frame_mutex_);
        frame_cv_.wait(ul, [this]() { return this->frame_ready_; });
        frame_ready_ = false;
        return frame_;
    }

    void Write(const std::shared_ptr<core::Tensor>& new_frame) {
        new_frame->AssertShape({frame_->GetShape()});
        new_frame->AssertDtype(frame_->GetDtype());

        // We use a one-directional (writer singals reader) signaling.
        // Removing this lock can results the unlikey event of missing signals,
        // which in our case it may be acceptable since the worst cenario is
        // a skipped frame. We kept the lock here just for safety.
        // https://stackoverflow.com/a/21439617/1255535
        std::unique_lock<std::mutex> ul(frame_mutex_);
        frame_ = new_frame;
        frame_ready_ = true;
        ul.unlock();
        frame_cv_.notify_one();
    }

private:
    GlobalBuffer() {
        core::Tensor two_fifty_five =
                core::Tensor::Ones({}, core::Dtype::UInt8) * 255;
        frame_ = std::make_shared<core::Tensor>(core::SizeVector({480, 640, 3}),
                                                core::Dtype::UInt8);
        frame_->Fill(0);
        frame_->Slice(0, 0, 160, 1).Slice(2, 0, 1, 1) = two_fifty_five;
        frame_->Slice(0, 160, 320, 1).Slice(2, 1, 2, 1) = two_fifty_five;
        frame_->Slice(0, 320, 480, 1).Slice(2, 2, 3, 1) = two_fifty_five;
    }

    virtual ~GlobalBuffer() {}

    std::shared_ptr<core::Tensor> frame_ = nullptr;
    std::mutex frame_mutex_;
    std::condition_variable frame_cv_;
    bool frame_ready_ = false;
    int num_initial_frames_ = 0;
};

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
