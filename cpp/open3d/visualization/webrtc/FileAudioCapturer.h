/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** RTSPAudioCapturer.h
**
** -------------------------------------------------------------------------*/

#pragma once

#include <live555helper/mkvclient.h>
#include <rtc_base/ref_counted_object.h>

#include "LiveAudioSource.h"

class FileAudioSource : public LiveAudioSource<MKVClient> {
public:
    static rtc::scoped_refptr<FileAudioSource> Create(
            rtc::scoped_refptr<webrtc::AudioDecoderFactory> audioDecoderFactory,
            const std::string& uri,
            const std::map<std::string, std::string>& opts) {
        rtc::scoped_refptr<FileAudioSource> source(
                new rtc::RefCountedObject<FileAudioSource>(audioDecoderFactory,
                                                           uri, opts));
        return source;
    }

protected:
    FileAudioSource(
            rtc::scoped_refptr<webrtc::AudioDecoderFactory> audioDecoderFactory,
            const std::string& uri,
            const std::map<std::string, std::string>& opts);
    virtual ~FileAudioSource();
};
