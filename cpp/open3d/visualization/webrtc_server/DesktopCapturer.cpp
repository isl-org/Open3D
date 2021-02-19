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
// ----------------------------------------------------------------------------
// Contains source code from
// https://github.com/mpromonet/webrtc-streamer
//
// This software is in the public domain, furnished "as is", without technical
// support, and with no warranty, express or implied, as to its usefulness for
// any purpose.
// ----------------------------------------------------------------------------
#ifdef USE_X11

#include "open3d/visualization/webrtc_server/DesktopCapturer.h"

#include <rtc_base/logging.h>

#include <fstream>
#include <iostream>

#include "open3d/core/Tensor.h"
#include "open3d/t/io/ImageIO.h"
#include "open3d/utility/Console.h"

namespace open3d {
namespace visualization {
namespace webrtc_server {

void DesktopCapturer::OnCaptureResult(
        webrtc::DesktopCapturer::Result result,
        std::unique_ptr<webrtc::DesktopFrame> frame) {
    RTC_LOG(INFO) << "DesktopCapturer:OnCaptureResult";

    if (result == webrtc::DesktopCapturer::Result::SUCCESS) {
        // Default
        // width: 640,
        // height: 480
        // kBytesPerPixel: 4,
        // frame->stride(): 2560
        int width = frame->stride() / webrtc::DesktopFrame::kBytesPerPixel;
        int height = frame->rect().height();
        // core::Tensor t_frame(static_cast<const uint8_t*>(frame->data()),
        //                      {height, width, 4}, core::Dtype::UInt8);
        // t_frame.Save("t_frame.npy");
        //
        // import numpy as np
        // import matplotlib.pyplot as plt
        // im = np.load("build/t_frame.npy")
        // im = np.flip(im[:, :, :3], axis=2)
        // print(im.shape)
        // print(im.dtype)
        // plt.imshow(im)
        //

        rtc::scoped_refptr<webrtc::I420Buffer> i420_buffer =
                webrtc::I420Buffer::Create(width, height);

        const int conversion_result = libyuv::ConvertToI420(
                frame->data(), 0, i420_buffer->MutableDataY(),
                i420_buffer->StrideY(), i420_buffer->MutableDataU(),
                i420_buffer->StrideU(), i420_buffer->MutableDataV(),
                i420_buffer->StrideV(), 0, 0, width, height,
                i420_buffer->width(), i420_buffer->height(), libyuv::kRotate0,
                ::libyuv::FOURCC_ARGB);

        if (conversion_result >= 0) {
            webrtc::VideoFrame video_frame(
                    i420_buffer, webrtc::VideoRotation::kVideoRotation_0,
                    rtc::TimeMicros());
            if ((height_ == 0) && (width_ == 0)) {
                broadcaster_.OnFrame(video_frame);

            } else {
                int height = height_;
                int width = width_;
                if (height == 0) {
                    height = (video_frame.height() * width) /
                             video_frame.width();
                } else if (width == 0) {
                    width = (video_frame.width() * height) /
                            video_frame.height();
                }
                int stride_y = width;
                int stride_uv = (width + 1) / 2;
                rtc::scoped_refptr<webrtc::I420Buffer> scaled_buffer =
                        webrtc::I420Buffer::Create(width, height, stride_y,
                                                   stride_uv, stride_uv);
                scaled_buffer->ScaleFrom(
                        *video_frame.video_frame_buffer()->ToI420());
                webrtc::VideoFrame frame = webrtc::VideoFrame(
                        scaled_buffer, webrtc::kVideoRotation_0,
                        rtc::TimeMicros());

                broadcaster_.OnFrame(frame);
            }
        } else {
            RTC_LOG(LS_ERROR)
                    << "DesktopCapturer:OnCaptureResult conversion error:"
                    << conversion_result;
        }

    } else {
        RTC_LOG(LS_ERROR) << "DesktopCapturer:OnCaptureResult capture error:"
                          << (int)result;
    }
}

void DesktopCapturer::CaptureThread() {
    RTC_LOG(INFO) << "DesktopCapturer:Run start";
    while (IsRunning()) {
        capturer_->CaptureFrame();
    }
    RTC_LOG(INFO) << "DesktopCapturer:Run exit";
}
bool DesktopCapturer::Start() {
    is_running_ = true;
    capture_thread_ = std::thread(&DesktopCapturer::CaptureThread, this);
    capturer_->Start(this);
    return true;
}

void DesktopCapturer::Stop() {
    is_running_ = false;
    capture_thread_.join();
}

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
#endif
