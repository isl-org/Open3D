/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** ScreenCapturer.h
**
** -------------------------------------------------------------------------*/

#pragma once

#include <api/video/i420_buffer.h>
#include <libyuv/convert.h>
#include <libyuv/video_common.h>
#include <media/base/video_broadcaster.h>
#include <media/base/video_common.h>
#include <modules/desktop_capture/desktop_capture_options.h>
#include <modules/desktop_capture/desktop_capturer.h>
#include <rtc_base/logging.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <thread>

#include "open3d/core/Tensor.h"
#include "open3d/t/io/ImageIO.h"
#include "open3d/utility/Console.h"

namespace open3d {
namespace visualization {
namespace webrtc_server {

class ImageReader {
public:
    ImageReader() {
        t::geometry::Image im;
        t::io::ReadImage(
                "/home/yixing/repo/Open3D/cpp/open3d/visualization/"
                "webrtc_server/html/lena_color_640_480.jpg",
                im);
        frame_ = core::Tensor::Zeros({im.GetRows(), im.GetCols(), 4},
                                     im.GetDtype());
        blank_ = core::Tensor::Zeros({im.GetRows(), im.GetCols(), 4},
                                     im.GetDtype());
        lena_ = core::Tensor::Zeros({im.GetRows(), im.GetCols(), 4},
                                    im.GetDtype());
        lena_.Slice(2, 0, 1) = im.AsTensor().Slice(2, 2, 3);
        lena_.Slice(2, 1, 2) = im.AsTensor().Slice(2, 1, 2);
        lena_.Slice(2, 2, 3) = im.AsTensor().Slice(2, 0, 1);
    }
    virtual ~ImageReader() {}

    class Callback {
    public:
        virtual void OnCaptureResult(const core::Tensor&) = 0;

    protected:
        virtual ~Callback() {}
    };

    void Start(Callback* callback) {
        utility::LogInfo("ImageReader::Start");
        callback_ = callback;
    }

    void CaptureFrame() {
        if (init_frame_count_ < 24) {
            init_frame_count_++;
            callback_->OnCaptureResult(frame_);
        } else {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        if (is_blank_) {
            frame_.AsRvalue() = lena_;
            is_blank_ = false;
        } else {
            frame_.AsRvalue() = blank_;
            is_blank_ = true;
        }
        callback_->OnCaptureResult(frame_);
    }

    Callback* callback_ = nullptr;

private:
    core::Tensor frame_;
    core::Tensor lena_;
    core::Tensor blank_;
    int init_frame_count_ = 0;
    bool is_blank_ = true;
};

class ImageCapturer : public rtc::VideoSourceInterface<webrtc::VideoFrame>,
                      public ImageReader::Callback {
public:
    ImageCapturer(const std::string& url_,
                  const std::map<std::string, std::string>& opts)
        : ImageCapturer(opts) {
        m_capturer = std::unique_ptr<ImageReader>(new ImageReader());
    }

    static ImageCapturer* Create(
            const std::string& url,
            const std::map<std::string, std::string>& opts) {
        std::unique_ptr<ImageCapturer> capturer(new ImageCapturer(url, opts));
        if (!capturer->Init()) {
            RTC_LOG(LS_WARNING) << "Failed to create ImageCapturer";
            return nullptr;
        }
        return capturer.release();
    }

    ImageCapturer(const std::map<std::string, std::string>& opts)
        : m_width(0), m_height(0) {
        if (opts.find("width") != opts.end()) {
            m_width = std::stoi(opts.at("width"));
        }
        if (opts.find("height") != opts.end()) {
            m_height = std::stoi(opts.at("height"));
        }
        // if (m_height != 480) {
        //     utility::LogError(
        //             "TODO: flexible height. Unsupported hight for now: {}",
        //             m_height);
        // }
        // if (m_width != 640) {
        //     utility::LogError(
        //             "TODO: flexible width. Unsupported width for now: {}",
        //             m_width);
        // }
    }
    bool Init() { return this->Start(); }
    virtual ~ImageCapturer() { this->Stop(); }

    void CaptureThread() {
        RTC_LOG(INFO) << "DesktopCapturer:Run start";
        while (IsRunning()) {
            m_capturer->CaptureFrame();
        }
        RTC_LOG(INFO) << "DesktopCapturer:Run exit";
    }

    bool Start() {
        m_capturer->Start(this);
        m_isrunning = true;
        m_capturethread = std::thread(&ImageCapturer::CaptureThread, this);
        return true;
    }
    void Stop() {
        m_isrunning = false;
        m_capturethread.join();
    }
    bool IsRunning() { return m_isrunning; }

    // overide webrtc::DesktopCapturer::Callback
    // See: WindowCapturerX11::CaptureFrame
    // build/webrtc/src/ext_webrtc/src/modules/desktop_capture/linux/window_capturer_x11.cc
    virtual void OnCaptureResult(const core::Tensor& frame) {
        utility::LogInfo("ImageCapturer:OnCaptureResult callback");
        int height = (int)frame.GetShape(0);
        int width = (int)frame.GetShape(1);

        rtc::scoped_refptr<webrtc::I420Buffer> I420buffer =
                webrtc::I420Buffer::Create(width, height);

        // frame->data()
        const int conversionResult = libyuv::ConvertToI420(
                static_cast<const uint8_t*>(frame.GetDataPtr()), 0,
                I420buffer->MutableDataY(), I420buffer->StrideY(),
                I420buffer->MutableDataU(), I420buffer->StrideU(),
                I420buffer->MutableDataV(), I420buffer->StrideV(), 0, 0, width,
                height, I420buffer->width(), I420buffer->height(),
                libyuv::kRotate0, ::libyuv::FOURCC_ARGB);

        if (conversionResult >= 0) {
            webrtc::VideoFrame videoFrame(
                    I420buffer, webrtc::VideoRotation::kVideoRotation_0,
                    rtc::TimeMicros());
            if ((m_height == 0) && (m_width == 0)) {
                broadcaster_.OnFrame(videoFrame);

            } else {
                int height = m_height;
                int width = m_width;
                if (height == 0) {
                    height = (videoFrame.height() * width) / videoFrame.width();
                } else if (width == 0) {
                    width = (videoFrame.width() * height) / videoFrame.height();
                }
                int stride_y = width;
                int stride_uv = (width + 1) / 2;
                rtc::scoped_refptr<webrtc::I420Buffer> scaled_buffer =
                        webrtc::I420Buffer::Create(width, height, stride_y,
                                                   stride_uv, stride_uv);
                scaled_buffer->ScaleFrom(
                        *videoFrame.video_frame_buffer()->ToI420());
                webrtc::VideoFrame frame = webrtc::VideoFrame(
                        scaled_buffer, webrtc::kVideoRotation_0,
                        rtc::TimeMicros());

                broadcaster_.OnFrame(frame);
            }
        } else {
            RTC_LOG(LS_ERROR)
                    << "DesktopCapturer:OnCaptureResult conversion error:"
                    << conversionResult;
        }
    }

    // overide rtc::VideoSourceInterface<webrtc::VideoFrame>
    virtual void AddOrUpdateSink(
            rtc::VideoSinkInterface<webrtc::VideoFrame>* sink,
            const rtc::VideoSinkWants& wants) {
        broadcaster_.AddOrUpdateSink(sink, wants);
    }

    virtual void RemoveSink(rtc::VideoSinkInterface<webrtc::VideoFrame>* sink) {
        broadcaster_.RemoveSink(sink);
    }

protected:
    std::thread m_capturethread;
    std::unique_ptr<ImageReader> m_capturer;
    int m_width;
    int m_height;
    bool m_isrunning;
    rtc::VideoBroadcaster broadcaster_;
};

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
