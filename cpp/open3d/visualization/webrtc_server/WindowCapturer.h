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

#include "open3d/visualization/webrtc_server/DesktopCapturer.h"

namespace open3d {
namespace visualization {
namespace webrtc_server {

class WindowCapturer : public DesktopCapturer {
public:
    WindowCapturer(const std::string& url,
                   const std::map<std::string, std::string>& opts)
        : DesktopCapturer(opts) {
        const std::string window_prefix("window://");
        if (url.find(window_prefix) == 0) {
            capturer_ = webrtc::DesktopCapturer::CreateWindowCapturer(
                    webrtc::DesktopCaptureOptions::CreateDefault());

            if (capturer_) {
                webrtc::DesktopCapturer::SourceList source_list;
                if (capturer_->GetSourceList(&source_list)) {
                    const std::string window_title(
                            url.substr(window_prefix.length()));
                    for (auto source : source_list) {
                        RTC_LOG(LS_ERROR)
                                << "WindowCapturer source:" << source.id
                                << " title:" << source.title;
                        if (window_title == source.title) {
                            capturer_->SelectSource(source.id);
                            break;
                        }
                    }
                }
            }
        }
    }
    static WindowCapturer* Create(
            const std::string& url,
            const std::map<std::string, std::string>& opts) {
        std::unique_ptr<WindowCapturer> capturer(new WindowCapturer(url, opts));
        if (!capturer->Init()) {
            RTC_LOG(LS_WARNING) << "Failed to create WindowCapturer";
            return nullptr;
        }
        return capturer.release();
    }
};

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
