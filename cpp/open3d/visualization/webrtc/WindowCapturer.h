/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** WindowCapturer.h
**
** -------------------------------------------------------------------------*/

#pragma once

#include "open3d/visualization/webrtc/DesktopCapturer.h"

class WindowCapturer : public DesktopCapturer {
public:
    WindowCapturer(const std::string& url,
                   const std::map<std::string, std::string>& opts)
        : DesktopCapturer(opts) {
        const std::string windowprefix("window://");
        if (url.find(windowprefix) == 0) {
            m_capturer = webrtc::DesktopCapturer::CreateWindowCapturer(
                    webrtc::DesktopCaptureOptions::CreateDefault());

            if (m_capturer) {
                webrtc::DesktopCapturer::SourceList sourceList;
                if (m_capturer->GetSourceList(&sourceList)) {
                    const std::string windowtitle(
                            url.substr(windowprefix.length()));
                    for (auto source : sourceList) {
                        RTC_LOG(LS_ERROR)
                                << "WindowCapturer source:" << source.id
                                << " title:" << source.title;
                        if (windowtitle == source.title) {
                            m_capturer->SelectSource(source.id);
                            break;
                        }
                    }
                }
            }
        }
    }
    static WindowCapturer* Create(
            const std::string& url,
            const std::map<std::string, std::string>& opts) {
        std::unique_ptr<WindowCapturer> capturer(new WindowCapturer(url, opts));
        if (!capturer->Init()) {
            RTC_LOG(LS_WARNING) << "Failed to create WindowCapturer";
            return nullptr;
        }
        return capturer.release();
    }
};
