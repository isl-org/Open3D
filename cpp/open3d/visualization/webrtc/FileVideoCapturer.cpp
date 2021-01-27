/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** filevideocapturer.cpp
**
** -------------------------------------------------------------------------*/

#ifdef HAVE_LIVE555

#include "open3d/visualization/webrtc/FileVideoCapturer.h"

#include <rtc_base/logging.h>

FileVideoCapturer::FileVideoCapturer(
        const std::string& uri, const std::map<std::string, std::string>& opts)
    : LiveVideoSource(uri, opts, true) {
    RTC_LOG(INFO) << "FileVideoCapturer " << uri;
}

FileVideoCapturer::~FileVideoCapturer() {}

#endif
