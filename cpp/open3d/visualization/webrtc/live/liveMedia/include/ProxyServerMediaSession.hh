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
// A subclass of "ServerMediaSession" that can be used to create a (unicast) RTSP servers that acts as a 'proxy' for
// another (unicast or multicast) RTSP/RTP stream.
// C++ header

#ifndef _PROXY_SERVER_MEDIA_SESSION_HH
#define _PROXY_SERVER_MEDIA_SESSION_HH

#ifndef _SERVER_MEDIA_SESSION_HH
#include "ServerMediaSession.hh"
#endif
#ifndef _MEDIA_SESSION_HH
#include "MediaSession.hh"
#endif
#ifndef _RTSP_CLIENT_HH
#include "RTSPClient.hh"
#endif
#ifndef _MEDIA_TRANSCODING_TABLE_HH
#include "MediaTranscodingTable.hh"
#endif

// A subclass of "RTSPClient", used to refer to the particular "ProxyServerMediaSession" object being used.
// It is used only within the implementation of "ProxyServerMediaSession", but is defined here, in case developers wish to
// subclass it.

class ProxyRTSPClient: public RTSPClient {
public:
  ProxyRTSPClient(class ProxyServerMediaSession& ourServerMediaSession, char const* rtspURL,
                  char const* username, char const* password,
                  portNumBits tunnelOverHTTPPortNum, int verbosityLevel, int socketNumToServer);
  virtual ~ProxyRTSPClient();

  void continueAfterDESCRIBE(char const* sdpDescription);
  void continueAfterLivenessCommand(int resultCode, Boolean serverSupportsGetParameter);
  void continueAfterSETUP(int resultCode);
  void continueAfterPLAY(int resultCode);
  void scheduleReset();

private:
  void reset();
  int connectToServer(int socketNum, portNumBits remotePortNum);

  Authenticator* auth() { return fOurAuthenticator; }

  void scheduleLivenessCommand();
  static void sendLivenessCommand(void* clientData);
  void doReset();
  static void doReset(void* clientData);

  void scheduleDESCRIBECommand();
  static void sendDESCRIBE(void* clientData);

  static void subsessionTimeout(void* clientData);
  void handleSubsessionTimeout();

private:
  friend class ProxyServerMediaSession;
  friend class ProxyServerMediaSubsession;
  ProxyServerMediaSession& fOurServerMediaSession;
  char* fOurURL;
  Authenticator* fOurAuthenticator;
  Boolean fStreamRTPOverTCP;
  class ProxyServerMediaSubsession *fSetupQueueHead, *fSetupQueueTail;
  unsigned fNumSetupsDone;
  unsigned fNextDESCRIBEDelay; // in seconds
  Boolean fServerSupportsGetParameter, fLastCommandWasPLAY, fDoneDESCRIBE;
  TaskToken fLivenessCommandTask, fDESCRIBECommandTask, fSubsessionTimerTask, fResetTask;
};


typedef ProxyRTSPClient*
createNewProxyRTSPClientFunc(ProxyServerMediaSession& ourServerMediaSession,
			     char const* rtspURL,
			     char const* username, char const* password,
			     portNumBits tunnelOverHTTPPortNum, int verbosityLevel,
			     int socketNumToServer);
ProxyRTSPClient*
defaultCreateNewProxyRTSPClientFunc(ProxyServerMediaSession& ourServerMediaSession,
				    char const* rtspURL,
				    char const* username, char const* password,
				    portNumBits tunnelOverHTTPPortNum, int verbosityLevel,
				    int socketNumToServer);

class ProxyServerMediaSession: public ServerMediaSession {
public:
  static ProxyServerMediaSession* createNew(UsageEnvironment& env,
					    GenericMediaServer* ourMediaServer, // Note: We can be used by just one server
					    char const* inputStreamURL, // the "rtsp://" URL of the stream we'll be proxying
					    char const* streamName = NULL,
					    char const* username = NULL, char const* password = NULL,
					    portNumBits tunnelOverHTTPPortNum = 0,
					        // for streaming the *proxied* (i.e., back-end) stream
					    int verbosityLevel = 0,
					    int socketNumToServer = -1,
					    MediaTranscodingTable* transcodingTable = NULL);
      // Hack: "tunnelOverHTTPPortNum" == 0xFFFF (i.e., all-ones) means: Stream RTP/RTCP-over-TCP, but *not* using HTTP
      // "verbosityLevel" == 1 means display basic proxy setup info; "verbosityLevel" == 2 means display RTSP client protocol also.
      // If "socketNumToServer" is >= 0, then it is the socket number of an already-existing TCP connection to the server.
      //      (In this case, "inputStreamURL" must point to the socket's endpoint, so that it can be accessed via the socket.)

  virtual ~ProxyServerMediaSession();

  char const* url() const;

  char describeCompletedFlag;
    // initialized to 0; set to 1 when the back-end "DESCRIBE" completes.
    // (This can be used as a 'watch variable' in "doEventLoop()".)
  Boolean describeCompletedSuccessfully() const { return fClientMediaSession != NULL; }
    // This can be used - along with "describeCompletdFlag" - to check whether the back-end "DESCRIBE" completed *successfully*.

protected:
  ProxyServerMediaSession(UsageEnvironment& env, GenericMediaServer* ourMediaServer,
			  char const* inputStreamURL, char const* streamName,
			  char const* username, char const* password,
			  portNumBits tunnelOverHTTPPortNum, int verbosityLevel,
			  int socketNumToServer,
			  MediaTranscodingTable* transcodingTable,
			  createNewProxyRTSPClientFunc* ourCreateNewProxyRTSPClientFunc
			  = defaultCreateNewProxyRTSPClientFunc,
			  portNumBits initialPortNum = 6970,
			  Boolean multiplexRTCPWithRTP = False);

  // If you subclass "ProxyRTSPClient", then you will also need to define your own function
  // - with signature "createNewProxyRTSPClientFunc" (see above) - that creates a new object
  // of this subclass.  You should also subclass "ProxyServerMediaSession" and, in your
  // subclass's constructor, initialize the parent class (i.e., "ProxyServerMediaSession")
  // constructor by passing your new function as the "ourCreateNewProxyRTSPClientFunc"
  // parameter.

  // Subclasses may redefine the following functions, if they want "ProxyServerSubsession"s
  // to create subclassed "Groupsock" and/or "RTCPInstance" objects:
  virtual Groupsock* createGroupsock(struct in_addr const& addr, Port port);
  virtual RTCPInstance* createRTCP(Groupsock* RTCPgs, unsigned totSessionBW, /* in kbps */
				   unsigned char const* cname, RTPSink* sink);

  virtual Boolean allowProxyingForSubsession(MediaSubsession const& mss);
  // By default, this function always returns True.  However, a subclass may redefine this
  // if it wishes to restrict which subsessions of a stream get proxied - e.g., if it wishes
  // to proxy only video tracks, but not audio (or other) tracks.

protected:
  GenericMediaServer* fOurMediaServer;
  ProxyRTSPClient* fProxyRTSPClient;
  MediaSession* fClientMediaSession;

private:
  friend class ProxyRTSPClient;
  friend class ProxyServerMediaSubsession;
  void continueAfterDESCRIBE(char const* sdpDescription);
  void resetDESCRIBEState(); // undoes what was done by "contineAfterDESCRIBE()"

private:
  int fVerbosityLevel;
  class PresentationTimeSessionNormalizer* fPresentationTimeSessionNormalizer;
  createNewProxyRTSPClientFunc* fCreateNewProxyRTSPClientFunc;
  MediaTranscodingTable* fTranscodingTable;
  portNumBits fInitialPortNum;
  Boolean fMultiplexRTCPWithRTP;
};


////////// PresentationTimeSessionNormalizer and PresentationTimeSubsessionNormalizer definitions //////////

// The following two classes are used by proxies to convert incoming streams' presentation times into wall-clock-aligned
// presentation times that are suitable for our "RTPSink"s (for the corresponding outgoing streams).
// (For multi-subsession (i.e., audio+video) sessions, the outgoing streams' presentation times retain the same relative
//  separation as those of the incoming streams.)

class PresentationTimeSubsessionNormalizer: public FramedFilter {
public:
  void setRTPSink(RTPSink* rtpSink) { fRTPSink = rtpSink; }

private:
  friend class PresentationTimeSessionNormalizer;
  PresentationTimeSubsessionNormalizer(PresentationTimeSessionNormalizer& parent, FramedSource* inputSource, RTPSource* rtpSource,
				       char const* codecName, PresentationTimeSubsessionNormalizer* next);
      // called only from within "PresentationTimeSessionNormalizer"
  virtual ~PresentationTimeSubsessionNormalizer();

  static void afterGettingFrame(void* clientData, unsigned frameSize,
                                unsigned numTruncatedBytes,
                                struct timeval presentationTime,
                                unsigned durationInMicroseconds);
  void afterGettingFrame(unsigned frameSize,
			 unsigned numTruncatedBytes,
			 struct timeval presentationTime,
			 unsigned durationInMicroseconds);

private: // redefined virtual functions:
  virtual void doGetNextFrame();

private:
  PresentationTimeSessionNormalizer& fParent;
  RTPSource* fRTPSource;
  RTPSink* fRTPSink;
  char const* fCodecName;
  PresentationTimeSubsessionNormalizer* fNext;
};

class PresentationTimeSessionNormalizer: public Medium {
public:
  PresentationTimeSessionNormalizer(UsageEnvironment& env);
  virtual ~PresentationTimeSessionNormalizer();

  PresentationTimeSubsessionNormalizer*
  createNewPresentationTimeSubsessionNormalizer(FramedSource* inputSource, RTPSource* rtpSource, char const* codecName);

private: // called only from within "~PresentationTimeSubsessionNormalizer":
  friend class PresentationTimeSubsessionNormalizer;
  void normalizePresentationTime(PresentationTimeSubsessionNormalizer* ssNormalizer,
				 struct timeval& toPT, struct timeval const& fromPT);
  void removePresentationTimeSubsessionNormalizer(PresentationTimeSubsessionNormalizer* ssNormalizer);

private:
  PresentationTimeSubsessionNormalizer* fSubsessionNormalizers;
  PresentationTimeSubsessionNormalizer* fMasterSSNormalizer; // used for subsessions that have been RTCP-synced

  struct timeval fPTAdjustment; // Added to (RTCP-synced) subsession presentation times to 'normalize' them with wall-clock time.
};

#endif
