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

#include <pc/video_track_source.h>

#include <regex>

#ifdef USE_X11
#include "open3d/visualization/webrtc_server/WindowCapturer.h"
#endif

#include "open3d/visualization/webrtc_server/ImageCapturer.h"

namespace open3d {
namespace visualization {
namespace webrtc_server {

template <class T>
class TrackSource : public webrtc::VideoTrackSource {
public:
    static rtc::scoped_refptr<TrackSource> Create(
            const std::string& video_url,
            const std::map<std::string, std::string>& opts) {
        std::unique_ptr<T> capturer =
                absl::WrapUnique(T::Create(video_url, opts));
        if (!capturer) {
            return nullptr;
        }
        return new rtc::RefCountedObject<TrackSource>(std::move(capturer));
    }

protected:
    explicit TrackSource(std::unique_ptr<T> capturer)
        : webrtc::VideoTrackSource(/*remote=*/false),
          capturer_(std::move(capturer)) {}

private:
    rtc::VideoSourceInterface<webrtc::VideoFrame>* source() override {
        return capturer_.get();
    }
    std::unique_ptr<T> capturer_;
};

class CapturerFactory {
public:
    static const std::list<std::string> GetVideoSourceList(
            const std::regex& publish_filter) {
        std::list<std::string> videoList;

#ifdef USE_X11
        if (std::regex_match("window://", publish_filter)) {
            std::unique_ptr<webrtc::DesktopCapturer> capturer =
                    webrtc::DesktopCapturer::CreateWindowCapturer(
                            webrtc::DesktopCaptureOptions::CreateDefault());
            if (capturer) {
                webrtc::DesktopCapturer::SourceList source_list;
                if (capturer->GetSourceList(&source_list)) {
                    for (auto source : source_list) {
                        std::ostringstream os;
                        os << "window://" << source.title;
                        videoList.push_back(os.str());
                    }
                }
            }
        }
#endif
        return videoList;
    }

    static rtc::scoped_refptr<webrtc::VideoTrackSourceInterface>
    CreateVideoSource(const std::string& video_url,
                      const std::map<std::string, std::string>& opts,
                      const std::regex& publish_filter,
                      rtc::scoped_refptr<webrtc::PeerConnectionFactoryInterface>
                              peer_connection_factory) {
        rtc::scoped_refptr<webrtc::VideoTrackSourceInterface> video_source;
        if ((video_url.find("window://") == 0) &&
            (std::regex_match("window://", publish_filter))) {
#ifdef USE_X11
            video_source = TrackSource<WindowCapturer>::Create(video_url, opts);
#endif
        } else if (video_url.find("image://") == 0) {
            video_source = TrackSource<ImageCapturer>::Create(video_url, opts);
        } else {
            utility::LogError("CreateVideoSource failed for video_url: {}",
                              video_url);
        }
        return video_source;
    }
};

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
