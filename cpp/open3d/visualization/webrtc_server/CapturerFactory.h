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

#include "open3d/visualization/webrtc_server/ImageCapturer.h"

namespace open3d {
namespace visualization {
namespace webrtc_server {

// class ImageCapturerTrackSource : public webrtc::VideoTrackSource {
// public:
//     static rtc::scoped_refptr<ImageCapturerTrackSource> Create(
//             const std::string& video_url,
//             const std::map<std::string, std::string>& opts) {
//         // TODO: remove this check after standarizing the track names.
//         if (video_url.find("image://") != 0) {
//             utility::LogError(
//                     "ImageCapturerTrackSource::Create failed for video_url:
//                     {}", video_url);
//         }
//         std::unique_ptr<ImageCapturer> capturer =
//                 absl::WrapUnique(ImageCapturer::Create(video_url, opts));
//         if (!capturer) {
//             return nullptr;
//         }
//         return new rtc::RefCountedObject<ImageCapturerTrackSource>(
//                 std::move(capturer));
//     }

// protected:
//     explicit ImageCapturerTrackSource(std::unique_ptr<ImageCapturer>
//     capturer)
//         : webrtc::VideoTrackSource(/*remote=*/false),
//           capturer_(std::move(capturer)) {}

// private:
//     rtc::VideoSourceInterface<webrtc::VideoFrame>* source() override {
//         return capturer_.get();
//     }
//     std::unique_ptr<ImageCapturer> capturer_;
// };

// class CapturerFactory {
// public:
//     static rtc::scoped_refptr<webrtc::VideoTrackSourceInterface>
//     CreateVideoSource(const std::string& video_url,
//                       const std::map<std::string, std::string>& opts) {
//         // TODO: remove this check later
//         if (video_url.find("image://") != 0) {
//             utility::LogError("CreateVideoSource failed for video_url: {}",
//                               video_url);
//         }

//         // rtc::scoped_refptr<webrtc::VideoTrackSourceInterface> video_source
//         =
//         //         ImageCapturerTrackSource::Create(video_url, opts);
//         return ImageCapturerTrackSource::Create(video_url, opts);
//     }
// };

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
