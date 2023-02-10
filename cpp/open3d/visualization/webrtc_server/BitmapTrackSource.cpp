// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
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
