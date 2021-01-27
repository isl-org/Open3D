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
// RTP sink for H.264 or H.265 video
// C++ header

#ifndef _H264_OR_5_VIDEO_RTP_SINK_HH
#define _H264_OR_5_VIDEO_RTP_SINK_HH

#ifndef _VIDEO_RTP_SINK_HH
#include "VideoRTPSink.hh"
#endif
#ifndef _FRAMED_FILTER_HH
#include "FramedFilter.hh"
#endif

class H264or5VideoRTPSink: public VideoRTPSink {
protected:
  H264or5VideoRTPSink(int hNumber, // 264 or 265
		      UsageEnvironment& env, Groupsock* RTPgs, unsigned char rtpPayloadFormat,
		      u_int8_t const* vps = NULL, unsigned vpsSize = 0,
		      u_int8_t const* sps = NULL, unsigned spsSize = 0,
		      u_int8_t const* pps = NULL, unsigned ppsSize = 0);
	// we're an abstrace base class
  virtual ~H264or5VideoRTPSink();

private: // redefined virtual functions:
  virtual Boolean continuePlaying();
  virtual void doSpecialFrameHandling(unsigned fragmentationOffset,
                                      unsigned char* frameStart,
                                      unsigned numBytesInFrame,
                                      struct timeval framePresentationTime,
                                      unsigned numRemainingBytes);
  virtual Boolean frameCanAppearAfterPacketStart(unsigned char const* frameStart,
						 unsigned numBytesInFrame) const;

protected:
  int fHNumber;
  FramedFilter* fOurFragmenter;
  char* fFmtpSDPLine;
  u_int8_t* fVPS; unsigned fVPSSize;
  u_int8_t* fSPS; unsigned fSPSSize;
  u_int8_t* fPPS; unsigned fPPSSize;
};

#endif
