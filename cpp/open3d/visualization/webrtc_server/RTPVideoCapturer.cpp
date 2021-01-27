/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** rtpvideocapturer.cpp
**
** -------------------------------------------------------------------------*/

#ifdef HAVE_LIVE555

#include "open3d/visualization/webrtc_server/RTPVideoCapturer.h"

#include <rtc_base/logging.h>

namespace open3d {
namespace visualization {
namespace webrtc_server {

RTPVideoCapturer::RTPVideoCapturer(
        const std::string& uri, const std::map<std::string, std::string>& opts)
    : LiveVideoSource(uri, opts, false) {
    RTC_LOG(INFO) << "RTSPVideoCapturer " << uri;
}

RTPVideoCapturer::~RTPVideoCapturer() {}

void RTPVideoCapturer::onError(SDPClient& connection, const char* error) {
    RTC_LOG(LS_ERROR) << "RTPVideoCapturer:onError error:" << error;
}
}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
#endif
