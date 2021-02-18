/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** VideoDecoder.h
**
** -------------------------------------------------------------------------*/

#pragma once

#include <api/video/i420_buffer.h>
#include <modules/video_coding/h264_sprop_parameter_sets.h>
#include <modules/video_coding/include/video_error_codes.h>

#include <cstring>
#include <vector>

namespace open3d {
namespace visualization {
namespace webrtc_server {

class VideoDecoder : public webrtc::DecodedImageCallback {
private:
    class Frame {
    public:
        Frame() : m_timestamp_ms(0) {}
        Frame(const rtc::scoped_refptr<webrtc::EncodedImageBuffer>& content,
              uint64_t timestamp_ms,
              webrtc::VideoFrameType frameType)
            : m_content(content),
              m_timestamp_ms(timestamp_ms),
              m_frameType(frameType) {}

        rtc::scoped_refptr<webrtc::EncodedImageBuffer> m_content;
        uint64_t m_timestamp_ms;
        webrtc::VideoFrameType m_frameType;
    };

public:
    VideoDecoder(rtc::VideoBroadcaster& broadcaster,
                 const std::map<std::string, std::string>& opts,
                 bool wait)
        : m_broadcaster(broadcaster),
          m_stop(false),
          m_wait(wait),
          m_previmagets(0),
          m_prevts(0) {}

    virtual ~VideoDecoder() {}

    void DecoderThread() {
        while (!m_stop) {
            std::unique_lock<std::mutex> mlock(m_queuemutex);
            while (m_queue.empty()) {
                m_queuecond.wait(mlock);
            }
            Frame frame = m_queue.front();
            m_queue.pop();
            mlock.unlock();

            if (frame.m_content.get() != nullptr) {
                RTC_LOG(LS_VERBOSE) << "VideoDecoder::DecoderThread size:"
                                    << frame.m_content->size()
                                    << " ts:" << frame.m_timestamp_ms;
                ssize_t size = frame.m_content->size();

                if (size) {
                    webrtc::EncodedImage input_image;
                    input_image.SetEncodedData(frame.m_content);
                    input_image._frameType = frame.m_frameType;
                    input_image.SetTimestamp(
                            frame.m_timestamp_ms);  // store time in ms that
                                                    // overflow the 32bits
                    int res = m_decoder->Decode(input_image, false,
                                                frame.m_timestamp_ms);
                    if (res != WEBRTC_VIDEO_CODEC_OK) {
                        RTC_LOG(LS_ERROR)
                                << "VideoDecoder::DecoderThread failure:"
                                << res;
                    }
                }
            }
        }
    }

    void Start() {
        RTC_LOG(INFO) << "VideoDecoder::start";
        m_stop = false;
        m_decoderthread = std::thread(&VideoDecoder::DecoderThread, this);
    }

    void Stop() {
        RTC_LOG(INFO) << "VideoDecoder::stop";
        m_stop = true;
        Frame frame;
        {
            std::unique_lock<std::mutex> lock(m_queuemutex);
            m_queue.push(frame);
        }
        m_queuecond.notify_all();
        m_decoderthread.join();
    }

    std::vector<std::vector<uint8_t>> getInitFrames(const std::string& codec,
                                                    const char* sdp) {
        std::vector<std::vector<uint8_t>> frames;

        if (codec == "H264") {
            const char* pattern = "sprop-parameter-sets=";
            const char* sprop = strstr(sdp, pattern);
            if (sprop) {
                std::string sdpstr(sprop + strlen(pattern));
                size_t pos = sdpstr.find_first_of(" ;\r\n");
                if (pos != std::string::npos) {
                    sdpstr.erase(pos);
                }
                webrtc::H264SpropParameterSets sprops;
                if (sprops.DecodeSprop(sdpstr)) {
                    std::vector<uint8_t> sps;
                    sps.insert(sps.end(), H26X_marker,
                               H26X_marker + sizeof(H26X_marker));
                    sps.insert(sps.end(), sprops.sps_nalu().begin(),
                               sprops.sps_nalu().end());
                    frames.push_back(sps);

                    std::vector<uint8_t> pps;
                    pps.insert(pps.end(), H26X_marker,
                               H26X_marker + sizeof(H26X_marker));
                    pps.insert(pps.end(), sprops.pps_nalu().begin(),
                               sprops.pps_nalu().end());
                    frames.push_back(pps);
                } else {
                    RTC_LOG(WARNING) << "Cannot decode SPS:" << sprop;
                }
            }
        }

        return frames;
    }

    void createDecoder(const std::string& codec) {
        webrtc::VideoCodec codec_settings;
        if (codec == "H264") {
            m_decoder = m_factory.CreateVideoDecoder(
                    webrtc::SdpVideoFormat(cricket::kH264CodecName));
            codec_settings.codecType = webrtc::VideoCodecType::kVideoCodecH264;
        } else if (codec == "VP9") {
            m_decoder = m_factory.CreateVideoDecoder(
                    webrtc::SdpVideoFormat(cricket::kVp9CodecName));
            codec_settings.codecType = webrtc::VideoCodecType::kVideoCodecVP9;
        }
        if (m_decoder.get() != nullptr) {
            m_decoder->InitDecode(&codec_settings, 2);
            m_decoder->RegisterDecodeCompleteCallback(this);
        }
    }

    void destroyDecoder() { m_decoder.reset(nullptr); }

    bool hasDecoder() { return (m_decoder.get() != nullptr); }

    void PostFrame(
            const rtc::scoped_refptr<webrtc::EncodedImageBuffer>& content,
            uint64_t ts,
            webrtc::VideoFrameType frameType) {
        Frame frame(content, ts, frameType);
        {
            std::unique_lock<std::mutex> lock(m_queuemutex);
            m_queue.push(frame);
        }
        m_queuecond.notify_all();
    }

    // overide webrtc::DecodedImageCallback
    virtual int32_t Decoded(webrtc::VideoFrame& decodedImage) override {
        int64_t ts = std::chrono::high_resolution_clock::now()
                             .time_since_epoch()
                             .count() /
                     1000 / 1000;

        RTC_LOG(LS_VERBOSE)
                << "VideoDecoder::Decoded size:" << decodedImage.size()
                << " decode ts:" << decodedImage.ntp_time_ms()
                << " source ts:" << ts;

        // waiting
        if ((m_wait) && (m_prevts != 0)) {
            int64_t periodSource = decodedImage.timestamp() - m_previmagets;
            int64_t periodDecode = ts - m_prevts;

            RTC_LOG(LS_VERBOSE) << "VideoDecoder::Decoded interframe decode:"
                                << periodDecode << " source:" << periodSource;
            int64_t delayms = periodSource - periodDecode;
            if ((delayms > 0) && (delayms < 1000)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(delayms));
            }
        }

        m_broadcaster.OnFrame(decodedImage);

        m_previmagets = decodedImage.timestamp();
        m_prevts = std::chrono::high_resolution_clock::now()
                           .time_since_epoch()
                           .count() /
                   1000 / 1000;

        return 1;
    }

    rtc::VideoBroadcaster& m_broadcaster;
    webrtc::InternalDecoderFactory m_factory;
    std::unique_ptr<webrtc::VideoDecoder> m_decoder;

    std::queue<Frame> m_queue;
    std::mutex m_queuemutex;
    std::condition_variable m_queuecond;
    std::thread m_decoderthread;
    bool m_stop;

    bool m_wait;
    int64_t m_previmagets;
    int64_t m_prevts;
};

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
