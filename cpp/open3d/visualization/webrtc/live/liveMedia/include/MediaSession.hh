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
// A data structure that represents a session that consists of
// potentially multiple (audio and/or video) sub-sessions
// (This data structure is used for media *receivers* - i.e., clients.
//  For media streamers, use "ServerMediaSession" instead.)
// C++ header

/* NOTE: To support receiving your own custom RTP payload format, you must first define a new
   subclass of "MultiFramedRTPSource" (or "BasicUDPSource") that implements it.
   Then define your own subclass of "MediaSession" and "MediaSubsession", as follows:
   - In your subclass of "MediaSession" (named, for example, "myMediaSession"):
       - Define and implement your own static member function
           static myMediaSession* createNew(UsageEnvironment& env, char const* sdpDescription);
	 and call this - instead of "MediaSession::createNew()" - in your application,
	 when you create a new "MediaSession" object.
       - Reimplement the "createNewMediaSubsession()" virtual function, as follows:
           MediaSubsession* myMediaSession::createNewMediaSubsession() { return new myMediaSubsession(*this); }
   - In your subclass of "MediaSubsession" (named, for example, "myMediaSubsession"):
       - Reimplement the "createSourceObjects()" virtual function, perhaps similar to this:
           Boolean myMediaSubsession::createSourceObjects(int useSpecialRTPoffset) {
	     if (strcmp(fCodecName, "X-MY-RTP-PAYLOAD-FORMAT") == 0) {
	       // This subsession uses our custom RTP payload format:
	       fReadSource = fRTPSource = myRTPPayloadFormatRTPSource::createNew( <parameters> );
	       return True;
	     } else {
	       // This subsession uses some other RTP payload format - perhaps one that we already implement:
	       return ::createSourceObjects(useSpecialRTPoffset);
	     }
	   }  
*/

#ifndef _MEDIA_SESSION_HH
#define _MEDIA_SESSION_HH

#ifndef _RTCP_HH
#include "RTCP.hh"
#endif
#ifndef _FRAMED_FILTER_HH
#include "FramedFilter.hh"
#endif

class MediaSubsession; // forward

class MediaSession: public Medium {
public:
  static MediaSession* createNew(UsageEnvironment& env,
				 char const* sdpDescription);

  static Boolean lookupByName(UsageEnvironment& env, char const* sourceName,
			      MediaSession*& resultSession);

  Boolean hasSubsessions() const { return fSubsessionsHead != NULL; }

  char* connectionEndpointName() const { return fConnectionEndpointName; }
  char const* CNAME() const { return fCNAME; }
  struct in_addr const& sourceFilterAddr() const { return fSourceFilterAddr; }
  float& scale() { return fScale; }
  float& speed() { return fSpeed; }
  char* mediaSessionType() const { return fMediaSessionType; }
  char* sessionName() const { return fSessionName; }
  char* sessionDescription() const { return fSessionDescription; }
  char const* controlPath() const { return fControlPath; }

  double& playStartTime() { return fMaxPlayStartTime; }
  double& playEndTime() { return fMaxPlayEndTime; }
  char* absStartTime() const;
  char* absEndTime() const;
  // Used only to set the local fields:
  char*& _absStartTime() { return fAbsStartTime; }
  char*& _absEndTime() { return fAbsEndTime; }

  Boolean initiateByMediaType(char const* mimeType,
			      MediaSubsession*& resultSubsession,
			      int useSpecialRTPoffset = -1);
      // Initiates the first subsession with the specified MIME type
      // Returns the resulting subsession, or 'multi source' (not both)

protected: // redefined virtual functions
  virtual Boolean isMediaSession() const;

protected:
  MediaSession(UsageEnvironment& env);
      // called only by createNew();
  virtual ~MediaSession();

  virtual MediaSubsession* createNewMediaSubsession();

  Boolean initializeWithSDP(char const* sdpDescription);
  Boolean parseSDPLine(char const* input, char const*& nextLine);
  Boolean parseSDPLine_s(char const* sdpLine);
  Boolean parseSDPLine_i(char const* sdpLine);
  Boolean parseSDPLine_c(char const* sdpLine);
  Boolean parseSDPAttribute_type(char const* sdpLine);
  Boolean parseSDPAttribute_control(char const* sdpLine);
  Boolean parseSDPAttribute_range(char const* sdpLine);
  Boolean parseSDPAttribute_source_filter(char const* sdpLine);

  static char* lookupPayloadFormat(unsigned char rtpPayloadType,
				   unsigned& rtpTimestampFrequency,
				   unsigned& numChannels);
  static unsigned guessRTPTimestampFrequency(char const* mediumName,
					     char const* codecName);

protected:
  friend class MediaSubsessionIterator;
  char* fCNAME; // used for RTCP

  // Linkage fields:
  MediaSubsession* fSubsessionsHead;
  MediaSubsession* fSubsessionsTail;

  // Fields set from a SDP description:
  char* fConnectionEndpointName;
  double fMaxPlayStartTime;
  double fMaxPlayEndTime;
  char* fAbsStartTime;
  char* fAbsEndTime;
  struct in_addr fSourceFilterAddr; // used for SSM
  float fScale; // set from a RTSP "Scale:" header
  float fSpeed;
  char* fMediaSessionType; // holds a=type value
  char* fSessionName; // holds s=<session name> value
  char* fSessionDescription; // holds i=<session description> value
  char* fControlPath; // holds optional a=control: string
};


class MediaSubsessionIterator {
public:
  MediaSubsessionIterator(MediaSession const& session);
  virtual ~MediaSubsessionIterator();

  MediaSubsession* next(); // NULL if none
  void reset();

private:
  MediaSession const& fOurSession;
  MediaSubsession* fNextPtr;
};


class MediaSubsession {
public:
  MediaSession& parentSession() { return fParent; }
  MediaSession const& parentSession() const { return fParent; }

  unsigned short clientPortNum() const { return fClientPortNum; }
  unsigned char rtpPayloadFormat() const { return fRTPPayloadFormat; }
  char const* savedSDPLines() const { return fSavedSDPLines; }
  char const* mediumName() const { return fMediumName; }
  char const* codecName() const { return fCodecName; }
  char const* protocolName() const { return fProtocolName; }
  char const* controlPath() const { return fControlPath; }
  Boolean isSSM() const { return fSourceFilterAddr.s_addr != 0; }

  unsigned short videoWidth() const { return fVideoWidth; }
  unsigned short videoHeight() const { return fVideoHeight; }
  unsigned videoFPS() const { return fVideoFPS; }
  unsigned numChannels() const { return fNumChannels; }
  float& scale() { return fScale; }
  float& speed() { return fSpeed; }

  RTPSource* rtpSource() { return fRTPSource; }
  RTCPInstance* rtcpInstance() { return fRTCPInstance; }
  unsigned rtpTimestampFrequency() const { return fRTPTimestampFrequency; }
  Boolean rtcpIsMuxed() const { return fMultiplexRTCPWithRTP; }
  FramedSource* readSource() { return fReadSource; }
    // This is the source that client sinks read from.  It is usually
    // (but not necessarily) the same as "rtpSource()"
  void addFilter(FramedFilter* filter);
    // Changes "readSource()" to "filter" (which must have just been created with "readSource()" as its input)

  double playStartTime() const;
  double playEndTime() const;
  char* absStartTime() const;
  char* absEndTime() const;
  // Used only to set the local fields:
  double& _playStartTime() { return fPlayStartTime; }
  double& _playEndTime() { return fPlayEndTime; }
  char*& _absStartTime() { return fAbsStartTime; }
  char*& _absEndTime() { return fAbsEndTime; }

  Boolean initiate(int useSpecialRTPoffset = -1);
      // Creates a "RTPSource" for this subsession. (Has no effect if it's
      // already been created.)  Returns True iff this succeeds.
  void deInitiate(); // Destroys any previously created RTPSource
  Boolean setClientPortNum(unsigned short portNum);
      // Sets the preferred client port number that any "RTPSource" for
      // this subsession would use.  (By default, the client port number
      // is gotten from the original SDP description, or - if the SDP
      // description does not specfy a client port number - an ephemeral
      // (even) port number is chosen.)  This routine must *not* be
      // called after initiate().
  void receiveRawMP3ADUs() { fReceiveRawMP3ADUs = True; } // optional hack for audio/MPA-ROBUST; must not be called after initiate()
  void receiveRawJPEGFrames() { fReceiveRawJPEGFrames = True; } // optional hack for video/JPEG; must not be called after initiate()
  char*& connectionEndpointName() { return fConnectionEndpointName; }
  char const* connectionEndpointName() const {
    return fConnectionEndpointName;
  }

  // 'Bandwidth' parameter, set in the "b=" SDP line:
  unsigned bandwidth() const { return fBandwidth; }

  // General SDP attribute accessor functions:
  char const* attrVal_str(char const* attrName) const;
      // returns "" if attribute doesn't exist (and has no default value), or is not a string
  char const* attrVal_strToLower(char const* attrName) const;
      // returns "" if attribute doesn't exist (and has no default value), or is not a string
  unsigned attrVal_int(char const* attrName) const;
      // also returns 0 if attribute doesn't exist (and has no default value)
  unsigned attrVal_unsigned(char const* attrName) const { return (unsigned)attrVal_int(attrName); }
  Boolean attrVal_bool(char const* attrName) const { return attrVal_int(attrName) != 0; }

  // Old, now-deprecated SDP attribute accessor functions, kept here for backwards-compatibility:
  char const* fmtp_config() const;
  char const* fmtp_configuration() const { return fmtp_config(); }
  char const* fmtp_spropparametersets() const { return attrVal_str("sprop-parameter-sets"); }
  char const* fmtp_spropvps() const { return attrVal_str("sprop-vps"); }
  char const* fmtp_spropsps() const { return attrVal_str("sprop-sps"); }
  char const* fmtp_sproppps() const { return attrVal_str("sprop-pps"); }

  netAddressBits connectionEndpointAddress() const;
      // Converts "fConnectionEndpointName" to an address (or 0 if unknown)
  void setDestinations(netAddressBits defaultDestAddress);
      // Uses "fConnectionEndpointName" and "serverPortNum" to set
      // the destination address and port of the RTP and RTCP objects.
      // This is typically called by RTSP clients after doing "SETUP".

  char const* sessionId() const { return fSessionId; }
  void setSessionId(char const* sessionId);

  // Public fields that external callers can use to keep state.
  // (They are responsible for all storage management on these fields)
  unsigned short serverPortNum; // in host byte order (used by RTSP)
  unsigned char rtpChannelId, rtcpChannelId; // used by RTSP (for RTP/TCP)
  MediaSink* sink; // callers can use this to keep track of who's playing us
  void* miscPtr; // callers can use this for whatever they want

  // Parameters set from a RTSP "RTP-Info:" header:
  struct {
    u_int16_t seqNum;
    u_int32_t timestamp;
    Boolean infoIsNew; // not part of the RTSP header; instead, set whenever this struct is filled in
  } rtpInfo;

  double getNormalPlayTime(struct timeval const& presentationTime);
  // Computes the stream's "Normal Play Time" (NPT) from the given "presentationTime".
  // (For the definition of "Normal Play Time", see RFC 2326, section 3.6.)
  // This function is useful only if the "rtpInfo" structure was previously filled in
  // (e.g., by a "RTP-Info:" header in a RTSP response).
  // Also, for this function to work properly, the RTP stream's presentation times must (eventually) be
  // synchronized via RTCP.
  // (Note: If this function returns a negative number, then the result should be ignored by the caller.)

protected:
  friend class MediaSession;
  friend class MediaSubsessionIterator;
  MediaSubsession(MediaSession& parent);
  virtual ~MediaSubsession();

  UsageEnvironment& env() { return fParent.envir(); }
  void setNext(MediaSubsession* next) { fNext = next; }

  void setAttribute(char const* name, char const* value = NULL, Boolean valueIsHexadecimal = False);

  Boolean parseSDPLine_c(char const* sdpLine);
  Boolean parseSDPLine_b(char const* sdpLine);
  Boolean parseSDPAttribute_rtpmap(char const* sdpLine);
  Boolean parseSDPAttribute_rtcpmux(char const* sdpLine);
  Boolean parseSDPAttribute_control(char const* sdpLine);
  Boolean parseSDPAttribute_range(char const* sdpLine);
  Boolean parseSDPAttribute_fmtp(char const* sdpLine);
  Boolean parseSDPAttribute_source_filter(char const* sdpLine);
  Boolean parseSDPAttribute_x_dimensions(char const* sdpLine);
  Boolean parseSDPAttribute_framerate(char const* sdpLine);

  virtual Boolean createSourceObjects(int useSpecialRTPoffset);
    // create "fRTPSource" and "fReadSource" member objects, after we've been initialized via SDP

protected:
  // Linkage fields:
  MediaSession& fParent;
  MediaSubsession* fNext;

  // Fields set from a SDP description:
  char* fConnectionEndpointName; // may also be set by RTSP SETUP response
  unsigned short fClientPortNum; // in host byte order
      // This field is also set by initiate()
  unsigned char fRTPPayloadFormat;
  char* fSavedSDPLines;
  char* fMediumName;
  char* fCodecName;
  char* fProtocolName;
  unsigned fRTPTimestampFrequency;
  Boolean fMultiplexRTCPWithRTP;
  char* fControlPath; // holds optional a=control: string
  struct in_addr fSourceFilterAddr; // used for SSM
  unsigned fBandwidth; // in kilobits-per-second, from b= line

  double fPlayStartTime;
  double fPlayEndTime;
  char* fAbsStartTime;
  char* fAbsEndTime;
  unsigned short fVideoWidth, fVideoHeight;
     // screen dimensions (set by an optional a=x-dimensions: <w>,<h> line)
  unsigned fVideoFPS;
     // frame rate (set by an optional "a=framerate: <fps>" or "a=x-framerate: <fps>" line)
  unsigned fNumChannels;
     // optionally set by "a=rtpmap:" lines for audio sessions.  Default: 1
  float fScale; // set from a RTSP "Scale:" header
  float fSpeed;
  double fNPT_PTS_Offset; // set by "getNormalPlayTime()"; add this to a PTS to get NPT
  HashTable* fAttributeTable; // for "a=fmtp:" attributes.  (Later an array by payload type #####)

  // Fields set or used by initiate():
  Groupsock* fRTPSocket; Groupsock* fRTCPSocket; // works even for unicast
  RTPSource* fRTPSource; RTCPInstance* fRTCPInstance;
  FramedSource* fReadSource;
  Boolean fReceiveRawMP3ADUs, fReceiveRawJPEGFrames;

  // Other fields:
  char* fSessionId; // used by RTSP
};

#endif
