/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** RTSPVideoCapturer.h
**
** -------------------------------------------------------------------------*/

#pragma once

#include <live555helper/rtspconnectionclient.h>

#include "open3d/visualization/webrtc/LiveVideoSource.h"

class RTSPVideoCapturer : public LiveVideoSource<RTSPConnection> {
public:
    RTSPVideoCapturer(const std::string& uri,
                      const std::map<std::string, std::string>& opts);
    virtual ~RTSPVideoCapturer();

    static RTSPVideoCapturer* Create(
            const std::string& url,
            const std::map<std::string, std::string>& opts) {
        return new RTSPVideoCapturer(url, opts);
    }

    // overide RTSPConnection::Callback
    virtual void onConnectionTimeout(RTSPConnection& connection) override {
        connection.start();
    }
    virtual void onDataTimeout(RTSPConnection& connection) override {
        connection.start();
    }
    virtual void onError(RTSPConnection& connection, const char* erro) override;
};
