/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** RTSPVideoCapturer.h
**
** -------------------------------------------------------------------------*/

#pragma once

#include <live555helper/sdpclient.h>

#include "LiveVideoSource.h"

class RTPVideoCapturer : public LiveVideoSource<SDPClient> {
public:
    RTPVideoCapturer(const std::string& uri,
                     const std::map<std::string, std::string>& opts);
    virtual ~RTPVideoCapturer();

    static RTPVideoCapturer* Create(
            const std::string& url,
            const std::map<std::string, std::string>& opts) {
        return new RTPVideoCapturer(url, opts);
    }

    // overide SDPClient::Callback
    virtual void onError(SDPClient& connection, const char* error) override;
};
