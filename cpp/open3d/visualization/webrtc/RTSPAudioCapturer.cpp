/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** rtspaudiocapturer.cpp
**
** -------------------------------------------------------------------------*/

#ifdef HAVE_LIVE555

#include "open3d/visualization/webrtc/RTSPAudioCapturer.h"

#include <rtc_base/logging.h>

RTSPAudioSource::RTSPAudioSource(
        rtc::scoped_refptr<webrtc::AudioDecoderFactory> audioDecoderFactory,
        const std::string& uri,
        const std::map<std::string, std::string>& opts)
    : LiveAudioSource(audioDecoderFactory, uri, opts, false) {}

RTSPAudioSource::~RTSPAudioSource() {}

#endif
