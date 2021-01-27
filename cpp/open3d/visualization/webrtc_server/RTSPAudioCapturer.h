/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** RTSPAudioCapturer.h
**
** -------------------------------------------------------------------------*/

#pragma once

#include <live555helper/rtspconnectionclient.h>
#include <rtc_base/ref_counted_object.h>

#include "open3d/visualization/webrtc_server/LiveAudioSource.h"

namespace open3d {
namespace visualization {
namespace webrtc_server {

class RTSPAudioSource : public LiveAudioSource<RTSPConnection> {
public:
    static rtc::scoped_refptr<RTSPAudioSource> Create(
            rtc::scoped_refptr<webrtc::AudioDecoderFactory> audioDecoderFactory,
            const std::string& uri,
            const std::map<std::string, std::string>& opts) {
        rtc::scoped_refptr<RTSPAudioSource> source(
                new rtc::RefCountedObject<RTSPAudioSource>(audioDecoderFactory,
                                                           uri, opts));
        return source;
    }

protected:
    RTSPAudioSource(
            rtc::scoped_refptr<webrtc::AudioDecoderFactory> audioDecoderFactory,
            const std::string& uri,
            const std::map<std::string, std::string>& opts);
    virtual ~RTSPAudioSource();
};

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
