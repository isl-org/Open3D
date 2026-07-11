// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/webrtc_server/ImageCapturer.h"

#include <api/scoped_refptr.h>
#include <api/video/color_space.h>
#include <api/video/i420_buffer.h>
#include <libyuv/convert.h>
#include <libyuv/video_common.h>
#include <media/base/video_broadcaster.h>
#include <media/base/video_common.h>
#include <rtc_base/time_utils.h>

#include <memory>

#include "open3d/core/Tensor.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace visualization {
namespace webrtc_server {

ImageCapturer::ImageCapturer(const std::string& url_,
                             const std::map<std::string, std::string>& opts)
    : ImageCapturer(opts) {}

ImageCapturer::~ImageCapturer() {}

ImageCapturer* ImageCapturer::Create(
        const std::string& url,
        const std::map<std::string, std::string>& opts) {
    std::unique_ptr<ImageCapturer> image_capturer(new ImageCapturer(url, opts));
    return image_capturer.release();
}

ImageCapturer::ImageCapturer(const std::map<std::string, std::string>& opts)
    : width_(0), height_(0) {
    if (opts.find("width") != opts.end()) {
        width_ = std::stoi(opts.at("width"));
    }
    if (opts.find("height") != opts.end()) {
        height_ = std::stoi(opts.at("height"));
    }
}

void ImageCapturer::OnCaptureResult(
        const std::shared_ptr<core::Tensor>& frame) {
    int height = (int)frame->GetShape(0);
    int width = (int)frame->GetShape(1);

    webrtc::scoped_refptr<webrtc::I420Buffer> i420_buffer =
            webrtc::I420Buffer::Create(width, height);

    // Use full-range ("J") conversion to match Open3D's full-range (0-255)
    // RGB frames, and tag the color space so VP9 signals this in-band.
    const int conversion_result = libyuv::RAWToJ420(
            frame->GetDataPtr<uint8_t>(), width * 3,
            i420_buffer->MutableDataY(), i420_buffer->StrideY(),
            i420_buffer->MutableDataU(), i420_buffer->StrideU(),
            i420_buffer->MutableDataV(), i420_buffer->StrideV(), width, height);

    if (conversion_result >= 0) {
        webrtc::VideoFrame video_frame(i420_buffer,
                                       webrtc::VideoRotation::kVideoRotation_0,
                                       webrtc::TimeMicros());
        video_frame.set_color_space(
                webrtc::ColorSpace(webrtc::ColorSpace::PrimaryID::kSMPTE170M,
                                   webrtc::ColorSpace::TransferID::kSMPTE170M,
                                   webrtc::ColorSpace::MatrixID::kSMPTE170M,
                                   webrtc::ColorSpace::RangeID::kFull));
        if ((height_ == 0) && (width_ == 0)) {
            broadcaster_.OnFrame(video_frame);
        } else {
            int height = height_;
            int width = width_;
            if (height == 0) {
                height = (video_frame.height() * width) / video_frame.width();
            } else if (width == 0) {
                width = (video_frame.width() * height) / video_frame.height();
            }
            int stride_y = width;
            int stride_uv = (width + 1) / 2;
            webrtc::scoped_refptr<webrtc::I420Buffer> scaled_buffer =
                    webrtc::I420Buffer::Create(width, height, stride_y,
                                               stride_uv, stride_uv);
            scaled_buffer->ScaleFrom(
                    *video_frame.video_frame_buffer()->ToI420());
            webrtc::VideoFrame frame =
                    webrtc::VideoFrame(scaled_buffer, webrtc::kVideoRotation_0,
                                       webrtc::TimeMicros());
            frame.set_color_space(*video_frame.color_space());

            broadcaster_.OnFrame(frame);
        }
    } else {
        utility::LogError("ImageCapturer:OnCaptureResult conversion error: {}",
                          conversion_result);
    }
}

// Override webrtc::VideoSourceInterface<webrtc::VideoFrame>.
void ImageCapturer::AddOrUpdateSink(
        webrtc::VideoSinkInterface<webrtc::VideoFrame>* sink,
        const webrtc::VideoSinkWants& wants) {
    broadcaster_.AddOrUpdateSink(sink, wants);
}

void ImageCapturer::RemoveSink(
        webrtc::VideoSinkInterface<webrtc::VideoFrame>* sink) {
    broadcaster_.RemoveSink(sink);
}

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
