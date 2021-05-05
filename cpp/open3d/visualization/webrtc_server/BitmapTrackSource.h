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
// Modified from: WebRTC src/pc/video_track_source.h
//
// Copyright 2016 The WebRTC project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// ----------------------------------------------------------------------------

#pragma once

#include <absl/types/optional.h>
#include <api/media_stream_interface.h>
#include <api/notifier.h>
#include <api/sequence_checker.h>
#include <api/video/recordable_encoded_frame.h>
#include <api/video/video_frame.h>
#include <api/video/video_sink_interface.h>
#include <api/video/video_source_interface.h>
#include <media/base/media_channel.h>

#include "open3d/core/Tensor.h"
#include "open3d/utility/Console.h"

namespace open3d {
namespace visualization {
namespace webrtc_server {

class BitmapTrackSourceInterface : public webrtc::VideoTrackSourceInterface {
public:
    virtual void OnFrame(const std::shared_ptr<core::Tensor>& frame) = 0;
};

// BitmapTrackSource is a convenience base class for implementations of
// VideoTrackSourceInterface.
class BitmapTrackSource : public webrtc::Notifier<BitmapTrackSourceInterface> {
public:
    explicit BitmapTrackSource(bool remote);
    void SetState(webrtc::MediaSourceInterface::SourceState new_state);

    webrtc::MediaSourceInterface::SourceState state() const override {
        return state_;
    }
    bool remote() const override { return remote_; }

    bool is_screencast() const override { return false; }
    absl::optional<bool> needs_denoising() const override {
        return absl::nullopt;
    }

    bool GetStats(Stats* stats) override { return false; }

    void AddOrUpdateSink(rtc::VideoSinkInterface<webrtc::VideoFrame>* sink,
                         const rtc::VideoSinkWants& wants) override;
    void RemoveSink(rtc::VideoSinkInterface<webrtc::VideoFrame>* sink) override;

    bool SupportsEncodedOutput() const override { return false; }
    void GenerateKeyFrame() override {}
    void AddEncodedSink(rtc::VideoSinkInterface<webrtc::RecordableEncodedFrame>*
                                sink) override {}
    void RemoveEncodedSink(
            rtc::VideoSinkInterface<webrtc::RecordableEncodedFrame>* sink)
            override {}

    // By default it does nothing (e.g. for VideoFilter).
    // ImageTrackSource overrides this and this will be called by the
    // BitmapWindowSystem when there's a new frame.
    virtual void OnFrame(const std::shared_ptr<core::Tensor>& frame) override {
        utility::LogInfo("BitmapTrackSource::OnFrame called");
    }

protected:
    virtual rtc::VideoSourceInterface<webrtc::VideoFrame>* source() = 0;

private:
    webrtc::SequenceChecker worker_thread_checker_;
    webrtc::MediaSourceInterface::SourceState state_;
    const bool remote_;
};

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
