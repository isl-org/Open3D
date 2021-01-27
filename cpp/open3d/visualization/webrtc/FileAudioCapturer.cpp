/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** fileaudiocapturer.cpp
**
** -------------------------------------------------------------------------*/

#ifdef HAVE_LIVE555

#include "FileAudioCapturer.h"

#include <rtc_base/logging.h>

FileAudioSource::FileAudioSource(
        rtc::scoped_refptr<webrtc::AudioDecoderFactory> audioDecoderFactory,
        const std::string& uri,
        const std::map<std::string, std::string>& opts)
    : LiveAudioSource(audioDecoderFactory, uri, opts, true) {
    RTC_LOG(INFO) << "FileAudioSource " << uri;
}

FileAudioSource::~FileAudioSource() {}
#endif
