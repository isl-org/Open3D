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

#pragma once

#include <api/video/i420_buffer.h>
#include <libyuv/convert.h>
#include <libyuv/video_common.h>
#include <media/base/video_broadcaster.h>
#include <media/base/video_common.h>
#include <modules/desktop_capture/desktop_capture_options.h>
#include <modules/desktop_capture/desktop_capturer.h>

#include <thread>

namespace open3d {
namespace visualization {
namespace webrtc_server {

class DesktopCapturer : public rtc::VideoSourceInterface<webrtc::VideoFrame>,
                        public webrtc::DesktopCapturer::Callback {
public:
    DesktopCapturer(const std::map<std::string, std::string>& opts)
        : width_(0), height_(0) {
        if (opts.find("width") != opts.end()) {
            width_ = std::stoi(opts.at("width"));
        }
        if (opts.find("height") != opts.end()) {
            height_ = std::stoi(opts.at("height"));
        }
    }
    bool Init() { return this->Start(); }
    virtual ~DesktopCapturer() { this->Stop(); }

    void CaptureThread();

    bool Start();
    void Stop();
    bool IsRunning() { return is_running_; }

    // overide webrtc::DesktopCapturer::Callback
    virtual void OnCaptureResult(webrtc::DesktopCapturer::Result result,
                                 std::unique_ptr<webrtc::DesktopFrame> frame);

    // overide rtc::VideoSourceInterface<webrtc::VideoFrame>
    virtual void AddOrUpdateSink(
            rtc::VideoSinkInterface<webrtc::VideoFrame>* sink,
            const rtc::VideoSinkWants& wants) {
        broadcaster_.AddOrUpdateSink(sink, wants);
    }

    virtual void RemoveSink(rtc::VideoSinkInterface<webrtc::VideoFrame>* sink) {
        broadcaster_.RemoveSink(sink);
    }

protected:
    std::thread capture_thread_;
    std::unique_ptr<webrtc::DesktopCapturer> capturer_;
    int width_;
    int height_;
    bool is_running_;
    rtc::VideoBroadcaster broadcaster_;
};

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
