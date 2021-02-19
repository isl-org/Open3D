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

namespace open3d {
namespace visualization {
namespace webrtc_server {

template <class T>
class VideoFilter : public webrtc::VideoTrackSource {
public:
    static rtc::scoped_refptr<VideoFilter> Create(
            rtc::scoped_refptr<webrtc::VideoTrackSourceInterface> video_source,
            const std::map<std::string, std::string>& opts) {
        std::unique_ptr<T> source = absl::WrapUnique(new T(video_source, opts));
        if (!source) {
            return nullptr;
        }
        return new rtc::RefCountedObject<VideoFilter>(std::move(source));
    }

protected:
    explicit VideoFilter(std::unique_ptr<T> source)
        : webrtc::VideoTrackSource(/*remote=*/false),
          source_(std::move(source)) {}

    SourceState state() const override { return kLive; }
    bool GetStats(Stats* stats) override {
        bool result = false;
        T* source = source_.get();
        if (source) {
            stats->input_height = source->height();
            stats->input_width = source->width();
            result = true;
        }
        return result;
    }

private:
    rtc::VideoSourceInterface<webrtc::VideoFrame>* source() override {
        return source_.get();
    }
    std::unique_ptr<T> source_;
};

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
