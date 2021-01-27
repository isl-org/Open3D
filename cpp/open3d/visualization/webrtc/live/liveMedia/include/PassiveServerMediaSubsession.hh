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
// A 'ServerMediaSubsession' object that represents an existing
// 'RTPSink', rather than one that creates new 'RTPSink's on demand.
// C++ header

#ifndef _PASSIVE_SERVER_MEDIA_SUBSESSION_HH
#define _PASSIVE_SERVER_MEDIA_SUBSESSION_HH

#ifndef _SERVER_MEDIA_SESSION_HH
#include "ServerMediaSession.hh"
#endif

#ifndef _RTP_SINK_HH
#include "RTPSink.hh"
#endif
#ifndef _RTCP_HH
#include "RTCP.hh"
#endif

class PassiveServerMediaSubsession: public ServerMediaSubsession {
public:
  static PassiveServerMediaSubsession* createNew(RTPSink& rtpSink,
						 RTCPInstance* rtcpInstance = NULL);

protected:
  PassiveServerMediaSubsession(RTPSink& rtpSink, RTCPInstance* rtcpInstance);
      // called only by createNew();
  virtual ~PassiveServerMediaSubsession();

  virtual Boolean rtcpIsMuxed();

protected: // redefined virtual functions
  virtual char const* sdpLines();
  virtual void getStreamParameters(unsigned clientSessionId,
				   netAddressBits clientAddress,
                                   Port const& clientRTPPort,
                                   Port const& clientRTCPPort,
				   int tcpSocketNum,
                                   unsigned char rtpChannelId,
                                   unsigned char rtcpChannelId,
                                   netAddressBits& destinationAddress,
				   u_int8_t& destinationTTL,
                                   Boolean& isMulticast,
                                   Port& serverRTPPort,
                                   Port& serverRTCPPort,
                                   void*& streamToken);
  virtual void startStream(unsigned clientSessionId, void* streamToken,
			   TaskFunc* rtcpRRHandler,
			   void* rtcpRRHandlerClientData,
                           unsigned short& rtpSeqNum,
                           unsigned& rtpTimestamp,
			   ServerRequestAlternativeByteHandler* serverRequestAlternativeByteHandler,
                           void* serverRequestAlternativeByteHandlerClientData);
  virtual float getCurrentNPT(void* streamToken);
  virtual void getRTPSinkandRTCP(void* streamToken,
				 RTPSink const*& rtpSink, RTCPInstance const*& rtcp);
  virtual void deleteStream(unsigned clientSessionId, void*& streamToken);

protected:
  char* fSDPLines;
  RTPSink& fRTPSink;
  RTCPInstance* fRTCPInstance;
  HashTable* fClientRTCPSourceRecords; // indexed by client session id; used to implement RTCP "RR" handling
};

#endif
