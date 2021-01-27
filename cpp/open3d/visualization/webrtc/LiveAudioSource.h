/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** LiveAudioSource.h
**
** -------------------------------------------------------------------------*/

#pragma once

#include <api/audio_codecs/builtin_audio_decoder_factory.h>
#include <live555helper/environment.h>
#include <pc/local_audio_source.h>
#include <rtc_base/logging.h>

#include <cctype>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

template <typename T>
class LiveAudioSource : public webrtc::Notifier<webrtc::AudioSourceInterface>,
                        public T::Callback {
public:
    SourceState state() const override { return kLive; }
    bool remote() const override { return true; }

    virtual void AddSink(webrtc::AudioTrackSinkInterface *sink) override {
        RTC_LOG(INFO) << "LiveAudioSource::AddSink ";
        std::lock_guard<std::mutex> lock(m_sink_lock);
        m_sinks.push_back(sink);
    }
    virtual void RemoveSink(webrtc::AudioTrackSinkInterface *sink) override {
        RTC_LOG(INFO) << "LiveAudioSource::RemoveSink ";
        std::lock_guard<std::mutex> lock(m_sink_lock);
        m_sinks.remove(sink);
    }

    void CaptureThread() { m_env.mainloop(); }

    // overide RTSPConnection::Callback
    virtual bool onNewSession(const char *id,
                              const char *media,
                              const char *codec,
                              const char *sdp) override {
        bool success = false;
        if (strcmp(media, "audio") == 0) {
            RTC_LOG(INFO) << "LiveAudioSource::onNewSession " << media << "/"
                          << codec << " " << sdp;

            // parse sdp to extract freq and channel
            std::string fmt(sdp);
            std::transform(fmt.begin(), fmt.end(), fmt.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            std::string codecstr(codec);
            std::transform(codecstr.begin(), codecstr.end(), codecstr.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            size_t pos = fmt.find(codecstr);
            if (pos != std::string::npos) {
                fmt.erase(0, pos + strlen(codec));
                fmt.erase(fmt.find_first_of(" \r\n"));
                std::istringstream is(fmt);
                std::string dummy;
                std::getline(is, dummy, '/');
                std::string freq;
                std::getline(is, freq, '/');
                if (!freq.empty()) {
                    m_freq = std::stoi(freq);
                }
                std::string channel;
                std::getline(is, channel, '/');
                if (!channel.empty()) {
                    m_channel = std::stoi(channel);
                }
            }
            RTC_LOG(INFO) << "LiveAudioSource::onNewSession codec:" << codecstr
                          << " freq:" << m_freq << " channel:" << m_channel;
            std::map<std::string, std::string> params;
            if (m_channel == 2) {
                params["stereo"] = "1";
            }

            webrtc::SdpAudioFormat format = webrtc::SdpAudioFormat(
                    codecstr, m_freq, m_channel, std::move(params));
            if (m_factory->IsSupportedDecoder(format)) {
                m_decoder = m_factory->MakeAudioDecoder(
                        format, absl::optional<webrtc::AudioCodecPairId>());
                m_codec[id] = codec;
                success = true;
            } else {
                RTC_LOG(LS_ERROR)
                        << "LiveAudioSource::onNewSession not support codec"
                        << sdp;
            }
        }
        return success;
    }
    virtual bool onData(const char *id,
                        unsigned char *buffer,
                        ssize_t size,
                        struct timeval presentationTime) override {
        bool success = false;
        int segmentLength = m_freq / 100;

        if (m_codec.find(id) != m_codec.end()) {
            int64_t sourcets = presentationTime.tv_sec;
            sourcets = sourcets * 1000 + presentationTime.tv_usec / 1000;

            int64_t ts = std::chrono::high_resolution_clock::now()
                                 .time_since_epoch()
                                 .count() /
                         1000 / 1000;

            RTC_LOG(LS_VERBOSE) << "LiveAudioSource::onData decode ts:" << ts
                                << " source ts:" << sourcets;

            if (m_decoder.get() != nullptr) {
                // waiting
                if ((m_wait) && (m_prevts != 0)) {
                    int64_t periodSource = sourcets - m_previmagets;
                    int64_t periodDecode = ts - m_prevts;

                    RTC_LOG(LS_VERBOSE)
                            << "LiveAudioSource::onData interframe decode:"
                            << periodDecode << " source:" << periodSource;
                    int64_t delayms = periodSource - periodDecode;
                    if ((delayms > 0) && (delayms < 1000)) {
                        std::this_thread::sleep_for(
                                std::chrono::milliseconds(delayms));
                    }
                }

                int maxDecodedBufferSize =
                        m_decoder->PacketDuration(buffer, size) * m_channel *
                        sizeof(int16_t);
                int16_t *decoded = new int16_t[maxDecodedBufferSize];
                webrtc::AudioDecoder::SpeechType speech_type;
                int decodedBufferSize = m_decoder->Decode(
                        buffer, size, m_freq, maxDecodedBufferSize, decoded,
                        &speech_type);
                RTC_LOG(LS_VERBOSE)
                        << "LiveAudioSource::onData size:" << size
                        << " decodedBufferSize:" << decodedBufferSize
                        << " maxDecodedBufferSize: " << maxDecodedBufferSize
                        << " channels: " << m_channel;
                if (decodedBufferSize > 0) {
                    for (int i = 0; i < decodedBufferSize; ++i) {
                        m_buffer.push(decoded[i]);
                    }
                } else {
                    RTC_LOG(LS_ERROR) << "LiveAudioSource::onData error:Decode "
                                         "Audio failed";
                }
                delete[] decoded;
                while ((int)m_buffer.size() > segmentLength * m_channel) {
                    int16_t *outbuffer = new int16_t[segmentLength * m_channel];
                    for (int i = 0; i < segmentLength * m_channel; ++i) {
                        uint16_t value = m_buffer.front();
                        outbuffer[i] = value;
                        m_buffer.pop();
                    }
                    std::lock_guard<std::mutex> lock(m_sink_lock);
                    for (auto *sink : m_sinks) {
                        sink->OnData(outbuffer, 16, m_freq, m_channel,
                                     segmentLength);
                    }
                    delete[] outbuffer;
                }

                m_previmagets = sourcets;
                m_prevts = std::chrono::high_resolution_clock::now()
                                   .time_since_epoch()
                                   .count() /
                           1000 / 1000;

                success = true;
            } else {
                RTC_LOG(LS_VERBOSE)
                        << "LiveAudioSource::onData error:No Audio decoder";
            }
        }
        return success;
    }

protected:
    LiveAudioSource(
            rtc::scoped_refptr<webrtc::AudioDecoderFactory> audioDecoderFactory,
            const std::string &uri,
            const std::map<std::string, std::string> &opts,
            bool wait)
        : m_env(m_stop),
          m_connection(m_env,
                       this,
                       uri.c_str(),
                       opts,
                       rtc::LogMessage::GetLogToDebug() <= 2),
          m_factory(audioDecoderFactory),
          m_freq(8000),
          m_channel(1),
          m_wait(wait),
          m_previmagets(0),
          m_prevts(0) {
        m_capturethread = std::thread(&LiveAudioSource::CaptureThread, this);
    }
    virtual ~LiveAudioSource() {
        m_env.stop();
        m_capturethread.join();
    }

private:
    char m_stop;
    Environment m_env;

private:
    T m_connection;
    std::thread m_capturethread;
    rtc::scoped_refptr<webrtc::AudioDecoderFactory> m_factory;
    std::unique_ptr<webrtc::AudioDecoder> m_decoder;
    int m_freq;
    int m_channel;
    std::queue<uint16_t> m_buffer;
    std::list<webrtc::AudioTrackSinkInterface *> m_sinks;
    std::mutex m_sink_lock;

    std::map<std::string, std::string> m_codec;

    bool m_wait;
    int64_t m_previmagets;
    int64_t m_prevts;
};
