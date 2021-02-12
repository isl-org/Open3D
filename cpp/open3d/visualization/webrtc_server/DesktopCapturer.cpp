/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** screencapturer.cpp
**
** -------------------------------------------------------------------------*/

#ifdef USE_X11

#include "open3d/visualization/webrtc_server/DesktopCapturer.h"

#include <rtc_base/logging.h>

#include <fstream>
#include <iostream>

#include "open3d/core/Tensor.h"
#include "open3d/t/io/ImageIO.h"
#include "open3d/utility/Console.h"

namespace open3d {
namespace visualization {
namespace webrtc_server {

void DesktopCapturer::OnCaptureResult(
        webrtc::DesktopCapturer::Result result,
        std::unique_ptr<webrtc::DesktopFrame> frame) {
    RTC_LOG(INFO) << "DesktopCapturer:OnCaptureResult";

    if (result == webrtc::DesktopCapturer::Result::SUCCESS) {
        // t::geometry::Image im;
        // t::io::ReadImage(
        //         "/home/yixing/repo/Open3D/cpp/open3d/visualization/"
        //         "webrtc_server/html/lena_color_640_480.jpg",
        //         im);
        // core::Tensor im_tensor = im.AsTensor();
        // core::Tensor im_tensor_bgra = core::Tensor::Zeros(
        //         {im.GetRows(), im.GetCols(), 4}, im_tensor.GetDtype());
        // im_tensor_bgra.Slice(2, 0, 1) = im_tensor.Slice(2, 2, 3);
        // im_tensor_bgra.Slice(2, 1, 2) = im_tensor.Slice(2, 1, 2);
        // im_tensor_bgra.Slice(2, 2, 3) = im_tensor.Slice(2, 0, 1);
        // set data to: static_cast<const uint8_t*>(im_tensor_bgra.GetDataPtr())

        // import numpy as np
        // import matplotlib.pyplot as plt
        // im = np.load("build/t_frame.npy")
        // im = np.flip(im[:, :, :3], axis=2)
        // print(im.shape)
        // print(im.dtype)
        // plt.imshow(im)
        int width = frame->stride() / webrtc::DesktopFrame::kBytesPerPixel;
        int height = frame->rect().height();
        // core::Tensor t_frame(static_cast<const uint8_t*>(frame->data()),
        //                      {height, width, 4}, core::Dtype::UInt8);
        // t_frame.Save("t_frame.npy");

        // width: 640,
        // height: 480
        // kBytesPerPixel: 4,
        // frame->stride(): 2560
        // utility::LogInfo(
        //         "width: {}, height: {}, kBytesPerPixel: {}, frame->stride():
        //         "
        //         "{}",
        //         width, height, webrtc::DesktopFrame::kBytesPerPixel,
        //         frame->stride());

        rtc::scoped_refptr<webrtc::I420Buffer> I420buffer =
                webrtc::I420Buffer::Create(width, height);

        const int conversionResult = libyuv::ConvertToI420(
                frame->data(), 0, I420buffer->MutableDataY(),
                I420buffer->StrideY(), I420buffer->MutableDataU(),
                I420buffer->StrideU(), I420buffer->MutableDataV(),
                I420buffer->StrideV(), 0, 0, width, height, I420buffer->width(),
                I420buffer->height(), libyuv::kRotate0, ::libyuv::FOURCC_ARGB);

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

    } else {
        RTC_LOG(LS_ERROR) << "DesktopCapturer:OnCaptureResult capture error:"
                          << (int)result;
    }
}

void DesktopCapturer::CaptureThread() {
    RTC_LOG(INFO) << "DesktopCapturer:Run start";
    while (IsRunning()) {
        m_capturer->CaptureFrame();
    }
    RTC_LOG(INFO) << "DesktopCapturer:Run exit";
}
bool DesktopCapturer::Start() {
    m_isrunning = true;
    m_capturethread = std::thread(&DesktopCapturer::CaptureThread, this);
    m_capturer->Start(this);
    return true;
}

void DesktopCapturer::Stop() {
    m_isrunning = false;
    m_capturethread.join();
}

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
#endif
