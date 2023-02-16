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

#include "open3d/visualization/webrtc_server/ImageCapturer.h"

#include <api/video/i420_buffer.h>
#include <libyuv/convert.h>
#include <libyuv/video_common.h>
#include <media/base/video_broadcaster.h>
#include <media/base/video_common.h>

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

    rtc::scoped_refptr<webrtc::I420Buffer> i420_buffer =
            webrtc::I420Buffer::Create(width, height);

    // frame->data()
    const int conversion_result = libyuv::ConvertToI420(
            frame->GetDataPtr<uint8_t>(), 0, i420_buffer->MutableDataY(),
            i420_buffer->StrideY(), i420_buffer->MutableDataU(),
            i420_buffer->StrideU(), i420_buffer->MutableDataV(),
            i420_buffer->StrideV(), 0, 0, width, height, i420_buffer->width(),
            i420_buffer->height(), libyuv::kRotate0, ::libyuv::FOURCC_RAW);

    if (conversion_result >= 0) {
        webrtc::VideoFrame video_frame(i420_buffer,
                                       webrtc::VideoRotation::kVideoRotation_0,
                                       rtc::TimeMicros());
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
            rtc::scoped_refptr<webrtc::I420Buffer> scaled_buffer =
                    webrtc::I420Buffer::Create(width, height, stride_y,
                                               stride_uv, stride_uv);
            scaled_buffer->ScaleFrom(
                    *video_frame.video_frame_buffer()->ToI420());
            webrtc::VideoFrame frame = webrtc::VideoFrame(
                    scaled_buffer, webrtc::kVideoRotation_0, rtc::TimeMicros());

            broadcaster_.OnFrame(frame);
        }
    } else {
        utility::LogError("ImageCapturer:OnCaptureResult conversion error: {}",
                          conversion_result);
    }
}

// Override rtc::VideoSourceInterface<webrtc::VideoFrame>.
void ImageCapturer::AddOrUpdateSink(
        rtc::VideoSinkInterface<webrtc::VideoFrame>* sink,
        const rtc::VideoSinkWants& wants) {
    broadcaster_.AddOrUpdateSink(sink, wants);
}

void ImageCapturer::RemoveSink(
        rtc::VideoSinkInterface<webrtc::VideoFrame>* sink) {
    broadcaster_.RemoveSink(sink);
}

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
