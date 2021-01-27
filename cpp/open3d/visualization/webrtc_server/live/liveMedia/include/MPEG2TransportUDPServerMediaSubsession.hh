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
// A 'ServerMediaSubsession' object that creates new, unicast, "RTPSink"s
// on demand, from an incoming UDP (or RTP/UDP) MPEG-2 Transport Stream
// C++ header

#ifndef _MPEG2_TRANSPORT_UDP_SERVER_MEDIA_SUBSESSION_HH
#define _MPEG2_TRANSPORT_UDP_SERVER_MEDIA_SUBSESSION_HH

#ifndef _ON_DEMAND_SERVER_MEDIA_SUBSESSION_HH
#include "OnDemandServerMediaSubsession.hh"
#endif

class MPEG2TransportUDPServerMediaSubsession: public OnDemandServerMediaSubsession {
public:
  static MPEG2TransportUDPServerMediaSubsession*
  createNew(UsageEnvironment& env,
	    char const* inputAddressStr, // An IP multicast address, or use "0.0.0.0" or NULL for unicast input
	    Port const& inputPort,
	    Boolean inputStreamIsRawUDP = False); // otherwise (default) the input stream is RTP/UDP
protected:
  MPEG2TransportUDPServerMediaSubsession(UsageEnvironment& env,
					 char const* inputAddressStr, Port const& inputPort, Boolean inputStreamIsRawUDP);
      // called only by createNew();
  virtual ~MPEG2TransportUDPServerMediaSubsession();

protected: // redefined virtual functions
  virtual FramedSource* createNewStreamSource(unsigned clientSessionId,
					      unsigned& estBitrate);
  virtual RTPSink* createNewRTPSink(Groupsock* rtpGroupsock,
				    unsigned char rtpPayloadTypeIfDynamic,
				    FramedSource* inputSource);
protected:
  char const* fInputAddressStr;
  Port fInputPort;
  Groupsock* fInputGroupsock;
  Boolean fInputStreamIsRawUDP;
};

#endif
