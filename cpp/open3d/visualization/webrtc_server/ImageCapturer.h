// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// This is a private header. It shall be hidden from Open3D's public API. Do not
// put this in Open3D.h.in.

#pragma once

#include <api/video/i420_buffer.h>
#include <libyuv/convert.h>
#include <libyuv/video_common.h>
#include <media/base/video_broadcaster.h>
#include <media/base/video_common.h>

#include <memory>

#include "open3d/core/Tensor.h"
#include "open3d/utility/Logging.h"
#include "open3d/visualization/webrtc_server/BitmapTrackSource.h"

namespace open3d {
namespace visualization {
namespace webrtc_server {

class ImageCapturer : public rtc::VideoSourceInterface<webrtc::VideoFrame> {
public:
    ImageCapturer(const std::string& url_,
                  const std::map<std::string, std::string>& opts);
    virtual ~ImageCapturer();

    static ImageCapturer* Create(
            const std::string& url,
            const std::map<std::string, std::string>& opts);

    ImageCapturer(const std::map<std::string, std::string>& opts);

    virtual void AddOrUpdateSink(
            rtc::VideoSinkInterface<webrtc::VideoFrame>* sink,
            const rtc::VideoSinkWants& wants) override;

    virtual void RemoveSink(
            rtc::VideoSinkInterface<webrtc::VideoFrame>* sink) override;

    void OnCaptureResult(const std::shared_ptr<core::Tensor>& frame);

protected:
    int width_;
    int height_;
    rtc::VideoBroadcaster broadcaster_;
};

class ImageTrackSource : public BitmapTrackSource {
public:
    static rtc::scoped_refptr<BitmapTrackSourceInterface> Create(
            const std::string& window_uid,
            const std::map<std::string, std::string>& opts) {
        std::unique_ptr<ImageCapturer> capturer =
                absl::WrapUnique(ImageCapturer::Create(window_uid, opts));
        if (!capturer) {
            return nullptr;
        }
        rtc::scoped_refptr<BitmapTrackSourceInterface> video_source =
                new rtc::RefCountedObject<ImageTrackSource>(
                        std::move(capturer));
        return video_source;
    }

    void OnFrame(const std::shared_ptr<core::Tensor>& frame) final override {
        capturer_->OnCaptureResult(frame);
    }

protected:
    explicit ImageTrackSource(std::unique_ptr<ImageCapturer> capturer)
        : BitmapTrackSource(/*remote=*/false), capturer_(std::move(capturer)) {}

private:
    rtc::VideoSourceInterface<webrtc::VideoFrame>* source() override {
        return capturer_.get();
    }
    std::unique_ptr<ImageCapturer> capturer_;
};

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
