/**********
This library is free software; you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the
Free Software Foundation; either version 3 of the License, or (at your
option) any later version. (See <http://www.gnu.org/copyleft/lesser.html>.)

This library is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for
more details.

You should have received a copy of the GNU Lesser General Public License
along with this library; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301  USA
**********/
// "liveMedia"
// Copyright (c) 1996-2020 Live Networks, Inc.  All rights reserved.
// A server that supports both RTSP, and HTTP streaming (using Apple's "HTTP Live Streaming" protocol)
// C++ header

#ifndef _RTSP_SERVER_SUPPORTING_HTTP_STREAMING_HH
#define _RTSP_SERVER_SUPPORTING_HTTP_STREAMING_HH

#ifndef _RTSP_SERVER_HH
#include "RTSPServer.hh"
#endif
#ifndef _BYTE_STREAM_MEMORY_BUFFER_SOURCE_HH
#include "ByteStreamMemoryBufferSource.hh"
#endif
#ifndef _TCP_STREAM_SINK_HH
#include "TCPStreamSink.hh"
#endif

class RTSPServerSupportingHTTPStreaming: public RTSPServer {
public:
  static RTSPServerSupportingHTTPStreaming* createNew(UsageEnvironment& env, Port rtspPort = 554,
						      UserAuthenticationDatabase* authDatabase = NULL,
						      unsigned reclamationTestSeconds = 65);

  Boolean setHTTPPort(Port httpPort) { return setUpTunnelingOverHTTP(httpPort); }

protected:
  RTSPServerSupportingHTTPStreaming(UsageEnvironment& env,
				    int ourSocket, Port ourPort,
				    UserAuthenticationDatabase* authDatabase,
				    unsigned reclamationTestSeconds);
      // called only by createNew();
  virtual ~RTSPServerSupportingHTTPStreaming();

protected: // redefined virtual functions
  virtual ClientConnection* createNewClientConnection(int clientSocket, struct sockaddr_in clientAddr);

public: // should be protected, but some old compilers complain otherwise
  class RTSPClientConnectionSupportingHTTPStreaming: public RTSPServer::RTSPClientConnection {
  public:
    RTSPClientConnectionSupportingHTTPStreaming(RTSPServer& ourServer, int clientSocket, struct sockaddr_in clientAddr);
    virtual ~RTSPClientConnectionSupportingHTTPStreaming();

  protected: // redefined virtual functions
    virtual void handleHTTPCmd_StreamingGET(char const* urlSuffix, char const* fullRequestStr);

  protected:
    static void afterStreaming(void* clientData);

  private:
    u_int32_t fClientSessionId;
    FramedSource* fStreamSource;
    ByteStreamMemoryBufferSource* fPlaylistSource;
    TCPStreamSink* fTCPSink;
  };
};

#endif
