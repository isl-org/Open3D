/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** sdpclient.h
**
** Interface to an SDP client
**
** -------------------------------------------------------------------------*/

#pragma once

#include <live555helper/environment.h>

#include <map>
#include <sstream>
#include <string>

#include "SessionSink.h"
#include "liveMedia.hh"

/* ---------------------------------------------------------------------------
**  SDP client connection interface
** -------------------------------------------------------------------------*/
class SDPClient {
public:
    static std::string getSdpFromRtpUrl(const std::string& url) {
        std::string sdp;
        if (url.find("rtp://") == 0) {
            std::istringstream is(url.substr(strlen("rtp://")));
            std::string ip;
            std::getline(is, ip, ':');
            std::string port;
            std::getline(is, port, '/');
            std::string rtppayloadtype("96");
            std::getline(is, rtppayloadtype, '/');
            std::string codec("H264");
            std::getline(is, codec);

            std::ostringstream os;
            os << "m=video " << port << " RTP/AVP " << rtppayloadtype << "\r\n"
               << "c=IN IP4 " << ip << "\r\n"
               << "a=rtpmap:" << rtppayloadtype << " " << codec << "\r\n";
            sdp.assign(os.str());
        }
        return sdp;
    }

    /* ---------------------------------------------------------------------------
    **  SDP client callback interface
    ** -------------------------------------------------------------------------*/
    class Callback : public SessionCallback {
    public:
        virtual void onError(SDPClient&, const char*) {}
    };

public:
    SDPClient(Environment& env,
              Callback* callback,
              const char* SDP,
              const std::map<std::string, std::string>& opts,
              int verbosityLevel = 1);
    virtual ~SDPClient();

protected:
    Environment& m_env;
    Callback* m_callback;
    MediaSession* m_session;
};
