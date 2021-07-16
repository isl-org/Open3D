// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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
// Modified from: WebRTC src/pc/video_track_source.cc
//
// Copyright 2016 The WebRTC project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// ----------------------------------------------------------------------------

#include "open3d/visualization/webrtc_server/BitmapTrackSource.h"

#include <pc/video_track_source.h>
#include <rtc_base/checks.h>

namespace open3d {
namespace visualization {
namespace webrtc_server {

BitmapTrackSource::BitmapTrackSource(bool remote)
    : state_(kInitializing), remote_(remote) {
    worker_thread_checker_.Detach();
}

void BitmapTrackSource::SetState(
        webrtc::MediaSourceInterface::SourceState new_state) {
    if (state_ != new_state) {
        state_ = new_state;
        FireOnChanged();
    }
}

void BitmapTrackSource::AddOrUpdateSink(
        rtc::VideoSinkInterface<webrtc::VideoFrame>* sink,
        const rtc::VideoSinkWants& wants) {
    RTC_DCHECK(worker_thread_checker_.IsCurrent());
    source()->AddOrUpdateSink(sink, wants);
}

void BitmapTrackSource::RemoveSink(
        rtc::VideoSinkInterface<webrtc::VideoFrame>* sink) {
    RTC_DCHECK(worker_thread_checker_.IsCurrent());
    source()->RemoveSink(sink);
}

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
