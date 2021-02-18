/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** CapturerFactory.h
**
** -------------------------------------------------------------------------*/

// # All sources
// DesktopCapturer.h    # X11 <- implicit
// WindowCapturer.h     # X11

#pragma once

#include <pc/video_track_source.h>

#include <regex>

#ifdef USE_X11
#include "open3d/visualization/webrtc_server/WindowCapturer.h"
#endif

#include "open3d/visualization/webrtc_server/ImageCapturer.h"

namespace open3d {
namespace visualization {
namespace webrtc_server {

template <class T>
class TrackSource : public webrtc::VideoTrackSource {
public:
    static rtc::scoped_refptr<TrackSource> Create(
            const std::string& videourl,
            const std::map<std::string, std::string>& opts) {
        std::unique_ptr<T> capturer =
                absl::WrapUnique(T::Create(videourl, opts));
        if (!capturer) {
            return nullptr;
        }
        return new rtc::RefCountedObject<TrackSource>(std::move(capturer));
    }

protected:
    explicit TrackSource(std::unique_ptr<T> capturer)
        : webrtc::VideoTrackSource(/*remote=*/false),
          capturer_(std::move(capturer)) {}

private:
    rtc::VideoSourceInterface<webrtc::VideoFrame>* source() override {
        return capturer_.get();
    }
    std::unique_ptr<T> capturer_;
};

class CapturerFactory {
public:
    static const std::list<std::string> GetVideoSourceList(
            const std::regex& publishFilter) {
        std::list<std::string> videoList;

#ifdef USE_X11
        if (std::regex_match("window://", publishFilter)) {
            std::unique_ptr<webrtc::DesktopCapturer> capturer =
                    webrtc::DesktopCapturer::CreateWindowCapturer(
                            webrtc::DesktopCaptureOptions::CreateDefault());
            if (capturer) {
                webrtc::DesktopCapturer::SourceList sourceList;
                if (capturer->GetSourceList(&sourceList)) {
                    for (auto source : sourceList) {
                        std::ostringstream os;
                        os << "window://" << source.title;
                        videoList.push_back(os.str());
                    }
                }
            }
        }
#endif
        return videoList;
    }

    static rtc::scoped_refptr<webrtc::VideoTrackSourceInterface>
    CreateVideoSource(const std::string& videourl,
                      const std::map<std::string, std::string>& opts,
                      const std::regex& publishFilter,
                      rtc::scoped_refptr<webrtc::PeerConnectionFactoryInterface>
                              peer_connection_factory) {
        rtc::scoped_refptr<webrtc::VideoTrackSourceInterface> videoSource;
        if ((videourl.find("window://") == 0) &&
            (std::regex_match("window://", publishFilter))) {
#ifdef USE_X11
            videoSource = TrackSource<WindowCapturer>::Create(videourl, opts);
#endif
        } else if (videourl.find("image://") == 0) {
            videoSource = TrackSource<ImageCapturer>::Create(videourl, opts);
        } else {
            utility::LogError("CreateVideoSource failed for videourl: {}",
                              videourl);
        }
        return videoSource;
    }

    static rtc::scoped_refptr<webrtc::AudioSourceInterface> CreateAudioSource(
            const std::string& audiourl,
            const std::map<std::string, std::string>& opts,
            const std::regex& publishFilter,
            rtc::scoped_refptr<webrtc::PeerConnectionFactoryInterface>
                    peer_connection_factory,
            rtc::scoped_refptr<webrtc::AudioDecoderFactory> audioDecoderfactory,
            rtc::scoped_refptr<webrtc::AudioDeviceModule> audioDeviceModule) {
        rtc::scoped_refptr<webrtc::AudioSourceInterface> audioSource;

        if (std::regex_match("audiocap://", publishFilter)) {
            audioDeviceModule->Init();
            int16_t num_audioDevices = audioDeviceModule->RecordingDevices();
            int16_t idx_audioDevice = -1;
            char name[webrtc::kAdmMaxDeviceNameSize] = {0};
            char id[webrtc::kAdmMaxGuidSize] = {0};
            if (audiourl.find("audiocap://") == 0) {
                int deviceNumber =
                        atoi(audiourl.substr(strlen("audiocap://")).c_str());
                RTC_LOG(INFO) << "audiourl:" << audiourl
                              << " device number:" << deviceNumber;
                if (audioDeviceModule->RecordingDeviceName(deviceNumber, name,
                                                           id) != -1) {
                    idx_audioDevice = deviceNumber;
                }

            } else {
                for (int i = 0; i < num_audioDevices; ++i) {
                    if (audioDeviceModule->RecordingDeviceName(i, name, id) !=
                        -1) {
                        RTC_LOG(INFO)
                                << "audiourl:" << audiourl
                                << " idx_audioDevice:" << i << " " << name;
                        if (audiourl == name) {
                            idx_audioDevice = i;
                            break;
                        }
                    }
                }
            }
            RTC_LOG(LS_ERROR) << "audiourl:" << audiourl
                              << " idx_audioDevice:" << idx_audioDevice << "/"
                              << num_audioDevices;
            if ((idx_audioDevice >= 0) &&
                (idx_audioDevice < num_audioDevices)) {
                audioDeviceModule->SetRecordingDevice(idx_audioDevice);
                cricket::AudioOptions opt;
                audioSource = peer_connection_factory->CreateAudioSource(opt);
            }
        } else {
            utility::LogError("CreateAudioSource failed for videourl: {}",
                              audiourl);
        }
        return audioSource;
    }
};

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
