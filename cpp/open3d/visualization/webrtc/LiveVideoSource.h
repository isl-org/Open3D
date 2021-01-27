/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** LiveVideoSource.h
**
** -------------------------------------------------------------------------*/

#pragma once

#include <api/video_codecs/video_decoder.h>
#include <common_video/h264/h264_common.h>
#include <common_video/h264/sps_parser.h>
#include <libyuv/convert.h>
#include <libyuv/video_common.h>
#include <live555helper/environment.h>
#include <media/base/codec.h>
#include <media/base/video_broadcaster.h>
#include <media/base/video_common.h>
#include <media/engine/internal_decoder_factory.h>
#include <modules/video_coding/h264_sprop_parameter_sets.h>
#include <rtc_base/logging.h>

#include <condition_variable>
#include <cstring>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "VideoDecoder.h"

template <typename T>
class LiveVideoSource : public rtc::VideoSourceInterface<webrtc::VideoFrame>,
                        public T::Callback {
public:
    LiveVideoSource(const std::string &uri,
                    const std::map<std::string, std::string> &opts,
                    bool wait)
        : m_env(m_stop),
          m_liveclient(m_env,
                       this,
                       uri.c_str(),
                       opts,
                       rtc::LogMessage::GetLogToDebug() <= 2),
          m_decoder(m_broadcaster, opts, wait) {
        this->Start();
    }
    virtual ~LiveVideoSource() { this->Stop(); }

    void Start() {
        RTC_LOG(INFO) << "LiveVideoSource::Start";
        m_capturethread = std::thread(&LiveVideoSource::CaptureThread, this);
        m_decoder.Start();
    }
    void Stop() {
        RTC_LOG(INFO) << "LiveVideoSource::stop";
        m_env.stop();
        m_capturethread.join();
        m_decoder.Stop();
    }
    bool IsRunning() { return (m_stop == 0); }

    void CaptureThread() { m_env.mainloop(); }

    // overide T::Callback
    virtual bool onNewSession(const char *id,
                              const char *media,
                              const char *codec,
                              const char *sdp) {
        bool success = false;
        if (strcmp(media, "video") == 0) {
            RTC_LOG(INFO) << "LiveVideoSource::onNewSession id:" << id
                          << " media:" << media << "/" << codec
                          << " sdp:" << sdp;

            if ((strcmp(codec, "H264") == 0) || (strcmp(codec, "JPEG") == 0) ||
                (strcmp(codec, "VP9") == 0)) {
                RTC_LOG(INFO) << "LiveVideoSource::onNewSession id:'" << id
                              << "' '" << codec << "'\n";
                m_codec[id] = codec;
                success = true;
            }
            RTC_LOG(INFO) << "LiveVideoSource::onNewSession success:" << success
                          << "\n";
            if (success) {
                struct timeval presentationTime;
                timerclear(&presentationTime);

                std::vector<std::vector<uint8_t>> initFrames =
                        m_decoder.getInitFrames(codec, sdp);
                for (auto frame : initFrames) {
                    onData(id, frame.data(), frame.size(), presentationTime);
                }
            }
        }
        return success;
    }
    virtual bool onData(const char *id,
                        unsigned char *buffer,
                        ssize_t size,
                        struct timeval presentationTime) {
        int64_t ts = presentationTime.tv_sec;
        ts = ts * 1000 + presentationTime.tv_usec / 1000;
        RTC_LOG(LS_VERBOSE) << "LiveVideoSource:onData id:" << id
                            << " size:" << size << " ts:" << ts;
        int res = 0;

        std::string codec = m_codec[id];
        RTC_LOG(LS_VERBOSE) << "LiveVideoSource: codec= " << m_codec[id];
        if (codec == "H264") {
            webrtc::H264::NaluType nalu_type =
                    webrtc::H264::ParseNaluType(buffer[sizeof(H26X_marker)]);
            if (nalu_type == webrtc::H264::NaluType::kSps) {
                RTC_LOG(LS_VERBOSE) << "LiveVideoSource:onData SPS";
                m_cfg.clear();
                m_cfg.insert(m_cfg.end(), buffer, buffer + size);

                absl::optional<webrtc::SpsParser::SpsState> sps =
                        webrtc::SpsParser::ParseSps(
                                buffer + sizeof(H26X_marker) +
                                        webrtc::H264::kNaluTypeSize,
                                size - sizeof(H26X_marker) -
                                        webrtc::H264::kNaluTypeSize);
                if (!sps) {
                    RTC_LOG(LS_ERROR) << "cannot parse sps";
                    res = -1;
                } else {
                    if (m_decoder.hasDecoder()) {
                        if ((m_format.width != (int)sps->width) ||
                            (m_format.height != (int)sps->height)) {
                            RTC_LOG(INFO)
                                    << "format changed => set format from "
                                    << m_format.width << "x" << m_format.height
                                    << " to " << sps->width << "x"
                                    << sps->height;
                            m_decoder.destroyDecoder();
                        }
                    }

                    if (!m_decoder.hasDecoder()) {
                        int fps = 25;
                        RTC_LOG(INFO)
                                << "LiveVideoSource:onData SPS set format "
                                << sps->width << "x" << sps->height
                                << " fps:" << fps;
                        cricket::VideoFormat videoFormat(
                                sps->width, sps->height,
                                cricket::VideoFormat::FpsToInterval(fps),
                                cricket::FOURCC_I420);
                        m_format = videoFormat;

                        m_decoder.createDecoder(codec);
                    }
                }
            } else if (nalu_type == webrtc::H264::NaluType::kPps) {
                RTC_LOG(LS_VERBOSE) << "LiveVideoSource:onData PPS";
                m_cfg.insert(m_cfg.end(), buffer, buffer + size);
            } else if (nalu_type == webrtc::H264::NaluType::kSei) {
            } else if (m_decoder.hasDecoder()) {
                webrtc::VideoFrameType frameType =
                        webrtc::VideoFrameType::kVideoFrameDelta;
                std::vector<uint8_t> content;
                if (nalu_type == webrtc::H264::NaluType::kIdr) {
                    frameType = webrtc::VideoFrameType::kVideoFrameKey;
                    RTC_LOG(LS_VERBOSE) << "LiveVideoSource:onData IDR";
                    content.insert(content.end(), m_cfg.begin(), m_cfg.end());
                } else {
                    RTC_LOG(LS_VERBOSE) << "LiveVideoSource:onData SLICE NALU:"
                                        << nalu_type;
                }
                content.insert(content.end(), buffer, buffer + size);
                rtc::scoped_refptr<webrtc::EncodedImageBuffer> frame =
                        webrtc::EncodedImageBuffer::Create(content.data(),
                                                           content.size());
                m_decoder.PostFrame(frame, ts, frameType);
            } else {
                RTC_LOG(LS_ERROR) << "LiveVideoSource:onData no decoder";
                res = -1;
            }
        } else if (codec == "JPEG") {
            int32_t width = 0;
            int32_t height = 0;
            if (libyuv::MJPGSize(buffer, size, &width, &height) == 0) {
                int stride_y = width;
                int stride_uv = (width + 1) / 2;

                rtc::scoped_refptr<webrtc::I420Buffer> I420buffer =
                        webrtc::I420Buffer::Create(width, height, stride_y,
                                                   stride_uv, stride_uv);
                const int conversionResult = libyuv::ConvertToI420(
                        (const uint8_t *)buffer, size,
                        I420buffer->MutableDataY(), I420buffer->StrideY(),
                        I420buffer->MutableDataU(), I420buffer->StrideU(),
                        I420buffer->MutableDataV(), I420buffer->StrideV(), 0, 0,
                        width, height, width, height, libyuv::kRotate0,
                        ::libyuv::FOURCC_MJPG);

                if (conversionResult >= 0) {
                    webrtc::VideoFrame frame =
                            webrtc::VideoFrame::Builder()
                                    .set_video_frame_buffer(I420buffer)
                                    .set_rotation(webrtc::kVideoRotation_0)
                                    .set_timestamp_ms(ts)
                                    .set_id(ts)
                                    .build();
                    m_decoder.Decoded(frame);
                } else {
                    RTC_LOG(LS_ERROR) << "LiveVideoSource:onData decoder error:"
                                      << conversionResult;
                    res = -1;
                }
            } else {
                RTC_LOG(LS_ERROR)
                        << "LiveVideoSource:onData cannot JPEG dimension";
                res = -1;
            }
        } else if (codec == "VP9") {
            if (!m_decoder.hasDecoder()) {
                m_decoder.createDecoder(codec);
            }
            if (m_decoder.hasDecoder()) {
                webrtc::VideoFrameType frameType =
                        webrtc::VideoFrameType::kVideoFrameKey;
                rtc::scoped_refptr<webrtc::EncodedImageBuffer> frame =
                        webrtc::EncodedImageBuffer::Create(buffer, size);
                m_decoder.PostFrame(frame, ts, frameType);
            }
        }

        return (res == 0);
    }

    // overide rtc::VideoSourceInterface<webrtc::VideoFrame>
    void AddOrUpdateSink(rtc::VideoSinkInterface<webrtc::VideoFrame> *sink,
                         const rtc::VideoSinkWants &wants) {
        m_broadcaster.AddOrUpdateSink(sink, wants);
    }

    void RemoveSink(rtc::VideoSinkInterface<webrtc::VideoFrame> *sink) {
        m_broadcaster.RemoveSink(sink);
    }

private:
    char m_stop;
    Environment m_env;

protected:
    T m_liveclient;

private:
    std::thread m_capturethread;
    cricket::VideoFormat m_format;
    std::vector<uint8_t> m_cfg;
    std::map<std::string, std::string> m_codec;

    rtc::VideoBroadcaster m_broadcaster;
    VideoDecoder m_decoder;
};
