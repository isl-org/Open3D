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

#include <api/media_stream_interface.h>
#include <media/base/video_broadcaster.h>

#include "open3d/visualization/webrtc_server/CustomVideoTrackSourceInterface.h"

namespace open3d {
namespace visualization {
namespace webrtc_server {

class VideoScaler : public rtc::VideoSinkInterface<webrtc::VideoFrame>,
                    public rtc::VideoSourceInterface<webrtc::VideoFrame> {
public:
    VideoScaler(
            rtc::scoped_refptr<webrtc::VideoTrackSourceInterface> video_source,
            const std::map<std::string, std::string> &opts)
        : video_source_(video_source),
          width_(0),
          height_(0),
          rotation_(webrtc::kVideoRotation_0),
          roi_x_(0),
          roi_y_(0),
          roi_width_(0),
          roi_height_(0) {
        if (opts.find("width") != opts.end()) {
            width_ = std::stoi(opts.at("width"));
        }
        if (opts.find("height") != opts.end()) {
            height_ = std::stoi(opts.at("height"));
        }
        if (opts.find("rotation") != opts.end()) {
            int rotation = std::stoi(opts.at("rotation"));
            switch (rotation) {
                case 90:
                    rotation_ = webrtc::kVideoRotation_90;
                    break;
                case 180:
                    rotation_ = webrtc::kVideoRotation_180;
                    break;
                case 270:
                    rotation_ = webrtc::kVideoRotation_270;
                    break;
            }
        }
        if (opts.find("roi_x") != opts.end()) {
            roi_x_ = std::stoi(opts.at("roi_x"));
            if (roi_x_ < 0) {
                RTC_LOG(LS_ERROR)
                        << "Ignore roi_x=" << roi_x_ << ", it muss be >=0";
                roi_x_ = 0;
            }
        }
        if (opts.find("roi_y") != opts.end()) {
            roi_y_ = std::stoi(opts.at("roi_y"));
            if (roi_y_ < 0) {
                RTC_LOG(LS_ERROR)
                        << "Ignore roi_<=" << roi_y_ << ", it muss be >=0";
                roi_y_ = 0;
            }
        }
        if (opts.find("roi_width") != opts.end()) {
            roi_width_ = std::stoi(opts.at("roi_width"));
            if (roi_width_ <= 0) {
                RTC_LOG(LS_ERROR) << "Ignore roi_width<=" << roi_width_
                                  << ", it muss be >0";
                roi_width_ = 0;
            }
        }
        if (opts.find("roi_height") != opts.end()) {
            roi_height_ = std::stoi(opts.at("roi_height"));
            if (roi_height_ <= 0) {
                RTC_LOG(LS_ERROR) << "Ignore roi_height<=" << roi_height_
                                  << ", it muss be >0";
                roi_height_ = 0;
            }
        }
    }

    virtual ~VideoScaler() {}

    void OnFrame(const webrtc::VideoFrame &frame) override {
        if (roi_x_ >= frame.width()) {
            RTC_LOG(LS_ERROR) << "The ROI position protrudes beyond the right "
                                 "edge of the image. Ignore roi_x.";
            roi_x_ = 0;
        }
        if (roi_y_ >= frame.height()) {
            RTC_LOG(LS_ERROR) << "The ROI position protrudes beyond the bottom "
                                 "edge of the image. Ignore roi_y.";
            roi_y_ = 0;
        }
        if (roi_width_ != 0 && (roi_width_ + roi_x_) > frame.width()) {
            RTC_LOG(LS_ERROR) << "The ROI protrudes beyond the right edge of "
                                 "the image. Ignore roi_width.";
            roi_width_ = 0;
        }
        if (roi_height_ != 0 && (roi_height_ + roi_y_) > frame.height()) {
            RTC_LOG(LS_ERROR) << "The ROI protrudes beyond the bottom edge of "
                                 "the image. Ignore roi_height.";
            roi_height_ = 0;
        }

        if (roi_width_ == 0) {
            roi_width_ = frame.width() - roi_x_;
        }
        if (roi_height_ == 0) {
            roi_height_ = frame.height() - roi_y_;
        }

        // source image is croped but destination image size is not set
        if ((roi_width_ != frame.width() || roi_height_ != frame.height()) &&
            (height_ == 0 && width_ == 0)) {
            height_ = roi_height_;
            width_ = roi_width_;
        }

        if ((height_ == 0) && (width_ == 0) &&
            (rotation_ == webrtc::kVideoRotation_0)) {
            broadcaster_.OnFrame(frame);
        } else {
            int height = height_;
            int width = width_;
            if ((height == 0) && (width == 0)) {
                height = frame.height();
                width = frame.width();
            } else if (height == 0) {
                height = (roi_height_ * width) / roi_width_;
            } else if (width == 0) {
                width = (roi_width_ * height) / roi_height_;
            }
            rtc::scoped_refptr<webrtc::I420Buffer> scaled_buffer =
                    webrtc::I420Buffer::Create(width, height);
            if (roi_width_ != frame.width() || roi_height_ != frame.height()) {
                scaled_buffer->CropAndScaleFrom(
                        *frame.video_frame_buffer()->ToI420(), roi_x_, roi_y_,
                        roi_width_, roi_height_);
            } else {
                scaled_buffer->ScaleFrom(*frame.video_frame_buffer()->ToI420());
            }
            webrtc::VideoFrame scaledFrame =
                    webrtc::VideoFrame(scaled_buffer, frame.timestamp(),
                                       frame.render_time_ms(), rotation_);

            broadcaster_.OnFrame(scaledFrame);
        }
    }

    void AddOrUpdateSink(rtc::VideoSinkInterface<webrtc::VideoFrame> *sink,
                         const rtc::VideoSinkWants &wants) override {
        video_source_->AddOrUpdateSink(this, wants);

        broadcaster_.AddOrUpdateSink(sink, wants);
    }

    void RemoveSink(
            rtc::VideoSinkInterface<webrtc::VideoFrame> *sink) override {
        video_source_->RemoveSink(this);

        broadcaster_.RemoveSink(sink);
    }

    int width() { return roi_width_; }
    int height() { return roi_height_; }

private:
    rtc::scoped_refptr<webrtc::VideoTrackSourceInterface> video_source_;
    rtc::VideoBroadcaster broadcaster_;

    int width_;
    int height_;
    webrtc::VideoRotation rotation_;
    int roi_x_;
    int roi_y_;
    int roi_width_;
    int roi_height_;
};

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
