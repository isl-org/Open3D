/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** filevideocapturer.cpp
**
** -------------------------------------------------------------------------*/

#ifdef HAVE_LIVE555

#include "open3d/visualization/webrtc_server/FileVideoCapturer.h"

#include <rtc_base/logging.h>

namespace open3d {
namespace visualization {
namespace webrtc_server {

FileVideoCapturer::FileVideoCapturer(
        const std::string& uri, const std::map<std::string, std::string>& opts)
    : LiveVideoSource(uri, opts, true) {
    RTC_LOG(INFO) << "FileVideoCapturer " << uri;
}

FileVideoCapturer::~FileVideoCapturer() {}
}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
#endif
