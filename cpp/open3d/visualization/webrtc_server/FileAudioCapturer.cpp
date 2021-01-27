/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** fileaudiocapturer.cpp
**
** -------------------------------------------------------------------------*/

#ifdef HAVE_LIVE555

#include "open3d/visualization/webrtc_server/FileAudioCapturer.h"

#include <rtc_base/logging.h>

namespace open3d {
namespace visualization {
namespace webrtc_server {

FileAudioSource::FileAudioSource(
        rtc::scoped_refptr<webrtc::AudioDecoderFactory> audioDecoderFactory,
        const std::string& uri,
        const std::map<std::string, std::string>& opts)
    : LiveAudioSource(audioDecoderFactory, uri, opts, true) {
    RTC_LOG(INFO) << "FileAudioSource " << uri;
}

FileAudioSource::~FileAudioSource() {}
}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
#endif
