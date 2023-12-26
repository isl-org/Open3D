// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// Contains source code from
// https://github.com/mpromonet/webrtc-streamer
//
// This software is in the public domain, furnished "as is", without technical
// support, and with no warranty, express or implied, as to its usefulness for
// any purpose.
// ----------------------------------------------------------------------------
//
// This is a private header. It shall be hidden from Open3D's public API. Do not
// put this in Open3D.h.in.

#pragma once

#include <pc/video_track_source.h>

#include "open3d/visualization/webrtc_server/BitmapTrackSource.h"

namespace open3d {
namespace visualization {
namespace webrtc_server {

/// \brief VideoFilter is a templated class for video frame processing.
///
/// VideoFilter is a BitmapTrackSource and it takes another BitmapTrackSource
/// as source and performs the video frame processing. The templated argument
/// implements the actual processing algorithm, e.g. VideoFilter<VideoScaler>.
template <class T>
class VideoFilter : public BitmapTrackSource {
public:
    static rtc::scoped_refptr<VideoFilter> Create(
            rtc::scoped_refptr<BitmapTrackSourceInterface> video_source,
            const std::map<std::string, std::string>& opts) {
        std::unique_ptr<T> source = absl::WrapUnique(new T(video_source, opts));
        if (!source) {
            return nullptr;
        }
        return new rtc::RefCountedObject<VideoFilter>(std::move(source));
    }

protected:
    explicit VideoFilter(std::unique_ptr<T> source)
        : BitmapTrackSource(/*remote=*/false), source_(std::move(source)) {}

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
