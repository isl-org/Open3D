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
// RTP sink for H.265 video
// C++ header

#ifndef _H265_VIDEO_RTP_SINK_HH
#define _H265_VIDEO_RTP_SINK_HH

#ifndef _H264_OR_5_VIDEO_RTP_SINK_HH
#include "H264or5VideoRTPSink.hh"
#endif

class H265VideoRTPSink: public H264or5VideoRTPSink {
public:
  static H265VideoRTPSink*
  createNew(UsageEnvironment& env, Groupsock* RTPgs, unsigned char rtpPayloadFormat);
  static H265VideoRTPSink*
  createNew(UsageEnvironment& env, Groupsock* RTPgs, unsigned char rtpPayloadFormat,
	    u_int8_t const* vps, unsigned vpsSize,
	    u_int8_t const* sps, unsigned spsSize,
	    u_int8_t const* pps, unsigned ppsSize);
    // an optional variant of "createNew()", useful if we know, in advance,
    // the stream's VPS, SPS and PPS NAL units.
    // This avoids us having to 'pre-read' from the input source in order to get these values.
  static H265VideoRTPSink*
  createNew(UsageEnvironment& env, Groupsock* RTPgs, unsigned char rtpPayloadFormat,
	    char const* sPropVPSStr, char const* sPropSPSStr, char const* sPropPPSStr);
    // an optional variant of "createNew()", useful if we know, in advance,
    // the stream's VPS, SPS and PPS NAL units.
    // This avoids us having to 'pre-read' from the input source in order to get these values.

protected:
  H265VideoRTPSink(UsageEnvironment& env, Groupsock* RTPgs, unsigned char rtpPayloadFormat,
		   u_int8_t const* vps = NULL, unsigned vpsSize = 0,
		   u_int8_t const* sps = NULL, unsigned spsSize = 0,
		   u_int8_t const* pps = NULL, unsigned ppsSize = 0);
	// called only by createNew()
  virtual ~H265VideoRTPSink();

protected: // redefined virtual functions:
  virtual char const* auxSDPLine();

private: // redefined virtual functions:
  virtual Boolean sourceIsCompatibleWithUs(MediaSource& source);
};

#endif
