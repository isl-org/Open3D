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
// on demand.
// C++ header

#ifndef _ON_DEMAND_SERVER_MEDIA_SUBSESSION_HH
#define _ON_DEMAND_SERVER_MEDIA_SUBSESSION_HH

#ifndef _SERVER_MEDIA_SESSION_HH
#include "ServerMediaSession.hh"
#endif
#ifndef _RTP_SINK_HH
#include "RTPSink.hh"
#endif
#ifndef _BASIC_UDP_SINK_HH
#include "BasicUDPSink.hh"
#endif
#ifndef _RTCP_HH
#include "RTCP.hh"
#endif

class OnDemandServerMediaSubsession: public ServerMediaSubsession {
protected: // we're a virtual base class
  OnDemandServerMediaSubsession(UsageEnvironment& env, Boolean reuseFirstSource,
				portNumBits initialPortNum = 6970,
				Boolean multiplexRTCPWithRTP = False);
  virtual ~OnDemandServerMediaSubsession();

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
  virtual void pauseStream(unsigned clientSessionId, void* streamToken);
  virtual void seekStream(unsigned clientSessionId, void* streamToken, double& seekNPT, double streamDuration, u_int64_t& numBytes);
  virtual void seekStream(unsigned clientSessionId, void* streamToken, char*& absStart, char*& absEnd);
  virtual void nullSeekStream(unsigned clientSessionId, void* streamToken,
			      double streamEndTime, u_int64_t& numBytes);
  virtual void setStreamScale(unsigned clientSessionId, void* streamToken, float scale);
  virtual float getCurrentNPT(void* streamToken);
  virtual FramedSource* getStreamSource(void* streamToken);
  virtual void getRTPSinkandRTCP(void* streamToken,
				 RTPSink const*& rtpSink, RTCPInstance const*& rtcp);
  virtual void deleteStream(unsigned clientSessionId, void*& streamToken);

protected: // new virtual functions, possibly redefined by subclasses
  virtual char const* getAuxSDPLine(RTPSink* rtpSink,
				    FramedSource* inputSource);
  virtual void seekStreamSource(FramedSource* inputSource, double& seekNPT, double streamDuration, u_int64_t& numBytes);
    // This routine is used to seek by relative (i.e., NPT) time.
    // "streamDuration", if >0.0, specifies how much data to stream, past "seekNPT".  (If <=0.0, all remaining data is streamed.)
    // "numBytes" returns the size (in bytes) of the data to be streamed, or 0 if unknown or unlimited.
  virtual void seekStreamSource(FramedSource* inputSource, char*& absStart, char*& absEnd);
    // This routine is used to seek by 'absolute' time.
    // "absStart" should be a string of the form "YYYYMMDDTHHMMSSZ" or "YYYYMMDDTHHMMSS.<frac>Z".
    // "absEnd" should be either NULL (for no end time), or a string of the same form as "absStart".
    // These strings may be modified in-place, or can be reassigned to a newly-allocated value (after delete[]ing the original).
  virtual void setStreamSourceScale(FramedSource* inputSource, float scale);
  virtual void setStreamSourceDuration(FramedSource* inputSource, double streamDuration, u_int64_t& numBytes);
  virtual void closeStreamSource(FramedSource* inputSource);

protected: // new virtual functions, defined by all subclasses
  virtual FramedSource* createNewStreamSource(unsigned clientSessionId,
					      unsigned& estBitrate) = 0;
      // "estBitrate" is the stream's estimated bitrate, in kbps
  virtual RTPSink* createNewRTPSink(Groupsock* rtpGroupsock,
				    unsigned char rtpPayloadTypeIfDynamic,
				    FramedSource* inputSource) = 0;

protected: // new virtual functions, may be redefined by a subclass:
  virtual Groupsock* createGroupsock(struct in_addr const& addr, Port port);
  virtual RTCPInstance* createRTCP(Groupsock* RTCPgs, unsigned totSessionBW, /* in kbps */
				   unsigned char const* cname, RTPSink* sink);

public:
  void multiplexRTCPWithRTP() { fMultiplexRTCPWithRTP = True; }
    // An alternative to passing the "multiplexRTCPWithRTP" parameter as True in the constructor

  void setRTCPAppPacketHandler(RTCPAppHandlerFunc* handler, void* clientData);
    // Sets a handler to be called if a RTCP "APP" packet arrives from any future client.
    // (Any current clients are not affected; any "APP" packets from them will continue to be
    // handled by whatever handler existed when the client sent its first RTSP "PLAY" command.)
    // (Call with (NULL, NULL) to remove an existing handler - for future clients only)

  void sendRTCPAppPacket(u_int8_t subtype, char const* name,
			 u_int8_t* appDependentData, unsigned appDependentDataSize);
    // Sends a custom RTCP "APP" packet to the most recent client (if "reuseFirstSource" was False),
    // or to all current clients (if "reuseFirstSource" was True).
    // The parameters correspond to their
    // respective fields as described in the RTP/RTCP definition (RFC 3550).
    // Note that only the low-order 5 bits of "subtype" are used, and only the first 4 bytes
    // of "name" are used.  (If "name" has fewer than 4 bytes, or is NULL,
    // then the remaining bytes are '\0'.)

protected:
  void setSDPLinesFromRTPSink(RTPSink* rtpSink, FramedSource* inputSource,
			      unsigned estBitrate);
      // used to implement "sdpLines()"

protected:
  char* fSDPLines;
  HashTable* fDestinationsHashTable; // indexed by client session id

private:
  Boolean fReuseFirstSource;
  portNumBits fInitialPortNum;
  Boolean fMultiplexRTCPWithRTP;
  void* fLastStreamToken;
  char fCNAME[100]; // for RTCP
  RTCPAppHandlerFunc* fAppHandlerTask;
  void* fAppHandlerClientData;
  friend class StreamState;
};


// A class that represents the state of an ongoing stream.  This is used only internally, in the implementation of
// "OnDemandServerMediaSubsession", but we expose the definition here, in case subclasses of "OnDemandServerMediaSubsession"
// want to access it.

class Destinations {
public:
  Destinations(struct in_addr const& destAddr,
               Port const& rtpDestPort,
               Port const& rtcpDestPort)
    : isTCP(False), addr(destAddr), rtpPort(rtpDestPort), rtcpPort(rtcpDestPort) {
  }
  Destinations(int tcpSockNum, unsigned char rtpChanId, unsigned char rtcpChanId)
    : isTCP(True), rtpPort(0) /*dummy*/, rtcpPort(0) /*dummy*/,
      tcpSocketNum(tcpSockNum), rtpChannelId(rtpChanId), rtcpChannelId(rtcpChanId) {
  }

public:
  Boolean isTCP;
  struct in_addr addr;
  Port rtpPort;
  Port rtcpPort;
  int tcpSocketNum;
  unsigned char rtpChannelId, rtcpChannelId;
};

class StreamState {
public:
  StreamState(OnDemandServerMediaSubsession& master,
              Port const& serverRTPPort, Port const& serverRTCPPort,
	      RTPSink* rtpSink, BasicUDPSink* udpSink,
	      unsigned totalBW, FramedSource* mediaSource,
	      Groupsock* rtpGS, Groupsock* rtcpGS);
  virtual ~StreamState();

  void startPlaying(Destinations* destinations, unsigned clientSessionId,
		    TaskFunc* rtcpRRHandler, void* rtcpRRHandlerClientData,
		    ServerRequestAlternativeByteHandler* serverRequestAlternativeByteHandler,
                    void* serverRequestAlternativeByteHandlerClientData);
  void pause();
  void sendRTCPAppPacket(u_int8_t subtype, char const* name,
			 u_int8_t* appDependentData, unsigned appDependentDataSize);
  void endPlaying(Destinations* destinations, unsigned clientSessionId);
  void reclaim();

  unsigned& referenceCount() { return fReferenceCount; }

  Port const& serverRTPPort() const { return fServerRTPPort; }
  Port const& serverRTCPPort() const { return fServerRTCPPort; }

  RTPSink* rtpSink() const { return fRTPSink; }
  RTCPInstance* rtcpInstance() const { return fRTCPInstance; }

  float streamDuration() const { return fStreamDuration; }

  FramedSource* mediaSource() const { return fMediaSource; }
  float& startNPT() { return fStartNPT; }

private:
  OnDemandServerMediaSubsession& fMaster;
  Boolean fAreCurrentlyPlaying;
  unsigned fReferenceCount;

  Port fServerRTPPort, fServerRTCPPort;

  RTPSink* fRTPSink;
  BasicUDPSink* fUDPSink;

  float fStreamDuration;
  unsigned fTotalBW;
  RTCPInstance* fRTCPInstance;

  FramedSource* fMediaSource;
  float fStartNPT; // initial 'normal play time'; reset after each seek

  Groupsock* fRTPgs;
  Groupsock* fRTCPgs;
};

#endif
