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
// RTCP
// C++ header

#ifndef _RTCP_HH
#define _RTCP_HH

#ifndef _RTP_SINK_HH
#include "RTPSink.hh"
#endif
#ifndef _RTP_SOURCE_HH
#include "RTPSource.hh"
#endif

class SDESItem {
public:
  SDESItem(unsigned char tag, unsigned char const* value);

  unsigned char const* data() const {return fData;}
  unsigned totalSize() const;

private:
  unsigned char fData[2 + 0xFF]; // first 2 bytes are tag and length
};

typedef void RTCPAppHandlerFunc(void* clientData,
				u_int8_t subtype, u_int32_t nameBytes/*big-endian order*/,
				u_int8_t* appDependentData, unsigned appDependentDataSize);

class RTCPMemberDatabase; // forward

typedef void ByeWithReasonHandlerFunc(void* clientData, char const* reason);

class RTCPInstance: public Medium {
public:
  static RTCPInstance* createNew(UsageEnvironment& env, Groupsock* RTCPgs,
				 unsigned totSessionBW, /* in kbps */
				 unsigned char const* cname,
				 RTPSink* sink,
				 RTPSource* source,
				 Boolean isSSMSource = False);

  static Boolean lookupByName(UsageEnvironment& env, char const* instanceName,
                              RTCPInstance*& resultInstance);

  unsigned numMembers() const;
  unsigned totSessionBW() const { return fTotSessionBW; }

  void setByeHandler(TaskFunc* handlerTask, void* clientData,
		     Boolean handleActiveParticipantsOnly = True);
      // Assigns a handler routine to be called if a "BYE" arrives.
      // The handler is called once only; for subsequent "BYE"s,
      // "setByeHandler()" would need to be called again.
      // If "handleActiveParticipantsOnly" is True, then the handler is called
      // only if the SSRC is for a known sender (if we have a "RTPSource"),
      // or if the SSRC is for a known receiver (if we have a "RTPSink").
      // This prevents (for example) the handler for a multicast receiver being
      // called if some other multicast receiver happens to exit.
      // If "handleActiveParticipantsOnly" is False, then the handler is called
      // for any incoming RTCP "BYE".
      // (To remove an existing "BYE" handler, call "setByeHandler()" again, with a "handlerTask" of NULL.)
  void setByeWithReasonHandler(ByeWithReasonHandlerFunc* handlerTask, void* clientData,
			       Boolean handleActiveParticipantsOnly = True);
      // Like "setByeHandler()", except that a string 'reason for the bye' (received as part of
      // the RTCP "BYE" packet) is passed to the handler function (along with "clientData").
      // (The 'reason' parameter to the handler function will be a dynamically-allocated string,
      // or NULL, and should be delete[]d by the handler function.)
  void setSRHandler(TaskFunc* handlerTask, void* clientData);
  void setRRHandler(TaskFunc* handlerTask, void* clientData);
      // Assigns a handler routine to be called if a "SR" or "RR" packet
      // (respectively) arrives.  Unlike "setByeHandler()", the handler will
      // be called once for each incoming "SR" or "RR".  (To turn off handling,
      // call the function again with "handlerTask" (and "clientData") as NULL.)
  void setSpecificRRHandler(netAddressBits fromAddress, Port fromPort,
			    TaskFunc* handlerTask, void* clientData);
      // Like "setRRHandler()", but applies only to "RR" packets that come from
      // a specific source address and port.  (Note that if both a specific
      // and a general "RR" handler function is set, then both will be called.)
  void unsetSpecificRRHandler(netAddressBits fromAddress, Port fromPort); // equivalent to setSpecificRRHandler(..., NULL, NULL);
  void setAppHandler(RTCPAppHandlerFunc* handlerTask, void* clientData);
      // Assigns a handler routine to be called whenever an "APP" packet arrives.  (To turn off
      // handling, call the function again with "handlerTask" (and "clientData") as NULL.)
  void sendAppPacket(u_int8_t subtype, char const* name,
		     u_int8_t* appDependentData, unsigned appDependentDataSize);
      // Sends a custom RTCP "APP" packet to the peer(s).  The parameters correspond to their
      // respective fields as described in the RTP/RTCP definition (RFC 3550).
      // Note that only the low-order 5 bits of "subtype" are used, and only the first 4 bytes
      // of "name" are used.  (If "name" has fewer than 4 bytes, or is NULL,
      // then the remaining bytes are '\0'.)

  Groupsock* RTCPgs() const { return fRTCPInterface.gs(); }

  void setStreamSocket(int sockNum, unsigned char streamChannelId);
  void addStreamSocket(int sockNum, unsigned char streamChannelId);
  void removeStreamSocket(int sockNum, unsigned char streamChannelId) {
    fRTCPInterface.removeStreamSocket(sockNum, streamChannelId);
  }
    // hacks to allow sending RTP over TCP (RFC 2236, section 10.12)

  void setAuxilliaryReadHandler(AuxHandlerFunc* handlerFunc,
                                void* handlerClientData) {
    fRTCPInterface.setAuxilliaryReadHandler(handlerFunc,
					    handlerClientData);
  }

  void injectReport(u_int8_t const* packet, unsigned packetSize, struct sockaddr_in const& fromAddress);
    // Allows an outside party to inject an RTCP report (from other than the network interface)

protected:
  RTCPInstance(UsageEnvironment& env, Groupsock* RTPgs, unsigned totSessionBW,
	       unsigned char const* cname,
	       RTPSink* sink, RTPSource* source,
	       Boolean isSSMSource);
      // called only by createNew()
  virtual ~RTCPInstance();

  virtual void noteArrivingRR(struct sockaddr_in const& fromAddressAndPort,
			      int tcpSocketNum, unsigned char tcpStreamChannelId);

  void incomingReportHandler1();

private:
  // redefined virtual functions:
  virtual Boolean isRTCPInstance() const;

private:
  Boolean addReport(Boolean alwaysAdd = False);
    void addSR();
    void addRR();
      void enqueueCommonReportPrefix(unsigned char packetType, u_int32_t SSRC,
				     unsigned numExtraWords = 0);
      void enqueueCommonReportSuffix();
        void enqueueReportBlock(RTPReceptionStats* receptionStats);
  void addSDES();
  void addBYE(char const* reason);

  void sendBuiltPacket();

  static void onExpire(RTCPInstance* instance);
  void onExpire1();

  static void incomingReportHandler(RTCPInstance* instance, int /*mask*/);
  void processIncomingReport(unsigned packetSize, struct sockaddr_in const& fromAddressAndPort,
			     int tcpSocketNum, unsigned char tcpStreamChannelId);
  void onReceive(int typeOfPacket, int totPacketSize, u_int32_t ssrc);

private:
  u_int8_t* fInBuf;
  unsigned fNumBytesAlreadyRead;
  OutPacketBuffer* fOutBuf;
  RTPInterface fRTCPInterface;
  unsigned fTotSessionBW;
  RTPSink* fSink;
  RTPSource* fSource;
  Boolean fIsSSMSource;

  SDESItem fCNAME;
  RTCPMemberDatabase* fKnownMembers;
  unsigned fOutgoingReportCount; // used for SSRC member aging

  double fAveRTCPSize;
  int fIsInitial;
  double fPrevReportTime;
  double fNextReportTime;
  int fPrevNumMembers;

  int fLastSentSize;
  int fLastReceivedSize;
  u_int32_t fLastReceivedSSRC;
  int fTypeOfEvent;
  int fTypeOfPacket;
  Boolean fHaveJustSentPacket;
  unsigned fLastPacketSentSize;

  TaskFunc* fByeHandlerTask;
  ByeWithReasonHandlerFunc* fByeWithReasonHandlerTask;
  void* fByeHandlerClientData;
  Boolean fByeHandleActiveParticipantsOnly;
  TaskFunc* fSRHandlerTask;
  void* fSRHandlerClientData;
  TaskFunc* fRRHandlerTask;
  void* fRRHandlerClientData;
  AddressPortLookupTable* fSpecificRRHandlerTable;
  RTCPAppHandlerFunc* fAppHandlerTask;
  void* fAppHandlerClientData;

public: // because this stuff is used by an external "C" function
  void schedule(double nextTime);
  void reschedule(double nextTime);
  void sendReport();
  void sendBYE(char const* reason = NULL);
  int typeOfEvent() {return fTypeOfEvent;}
  int sentPacketSize() {return fLastSentSize;}
  int packetType() {return fTypeOfPacket;}
  int receivedPacketSize() {return fLastReceivedSize;}
  int checkNewSSRC();
  void removeLastReceivedSSRC();
  void removeSSRC(u_int32_t ssrc, Boolean alsoRemoveStats);
};

// RTCP packet types:
const unsigned char RTCP_PT_SR = 200;
const unsigned char RTCP_PT_RR = 201;
const unsigned char RTCP_PT_SDES = 202;
const unsigned char RTCP_PT_BYE = 203;
const unsigned char RTCP_PT_APP = 204;
const unsigned char RTCP_PT_RTPFB = 205; // Generic RTP Feedback [RFC4585]
const unsigned char RTCP_PT_PSFB = 206; // Payload-specific [RFC4585]
const unsigned char RTCP_PT_XR = 207; // extended report [RFC3611]
const unsigned char RTCP_PT_AVB = 208; // AVB RTCP packet ["Standard for Layer 3 Transport Protocol for Time Sensitive Applications in Local Area Networks." Work in progress.]
const unsigned char RTCP_PT_RSI = 209; // Receiver Summary Information [RFC5760]
const unsigned char RTCP_PT_TOKEN = 210; // Port Mapping [RFC6284]
const unsigned char RTCP_PT_IDMS = 211; // IDMS Settings [RFC7272]

// SDES tags:
const unsigned char RTCP_SDES_END = 0;
const unsigned char RTCP_SDES_CNAME = 1;
const unsigned char RTCP_SDES_NAME = 2;
const unsigned char RTCP_SDES_EMAIL = 3;
const unsigned char RTCP_SDES_PHONE = 4;
const unsigned char RTCP_SDES_LOC = 5;
const unsigned char RTCP_SDES_TOOL = 6;
const unsigned char RTCP_SDES_NOTE = 7;
const unsigned char RTCP_SDES_PRIV = 8;

#endif
