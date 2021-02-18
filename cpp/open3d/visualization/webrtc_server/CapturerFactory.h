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
};

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
