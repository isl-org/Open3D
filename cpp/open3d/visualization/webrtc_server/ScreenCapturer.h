/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** ScreenCapturer.h
**
** -------------------------------------------------------------------------*/

#pragma once

#include "open3d/visualization/webrtc_server/DesktopCapturer.h"

namespace open3d {
namespace visualization {
namespace webrtc_server {

class ScreenCapturer : public DesktopCapturer {
public:
    ScreenCapturer(const std::string& url,
                   const std::map<std::string, std::string>& opts)
        : DesktopCapturer(opts) {
        const std::string prefix("screen://");
        m_capturer = webrtc::DesktopCapturer::CreateScreenCapturer(
                webrtc::DesktopCaptureOptions::CreateDefault());
        if (m_capturer) {
            webrtc::DesktopCapturer::SourceList sourceList;
            if (m_capturer->GetSourceList(&sourceList)) {
                const std::string screen(url.substr(prefix.length()));
                if (screen.empty() == false) {
                    for (auto source : sourceList) {
                        RTC_LOG(LS_ERROR)
                                << "ScreenCapturer source:" << source.id
                                << " title:" << source.title;
                        if (atoi(screen.c_str()) == source.id) {
                            m_capturer->SelectSource(source.id);
                            break;
                        }
                    }
                }
            }
        }
    }
    static ScreenCapturer* Create(
            const std::string& url,
            const std::map<std::string, std::string>& opts) {
        std::unique_ptr<ScreenCapturer> capturer(new ScreenCapturer(url, opts));
        if (!capturer->Init()) {
            RTC_LOG(LS_WARNING) << "Failed to create WindowCapturer";
            return nullptr;
        }
        return capturer.release();
    }
};
}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
