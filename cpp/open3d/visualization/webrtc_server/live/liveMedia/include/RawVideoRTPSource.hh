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
// Raw Video RTP Sources (RFC 4175)
// C++ header

#ifndef _RAW_VIDEO_RTP_SOURCE_HH
#define _RAW_VIDEO_RTP_SOURCE_HH

#ifndef _MULTI_FRAMED_RTP_SOURCE_HH
#include "MultiFramedRTPSource.hh"
#endif

class RawVideoRTPSource: public MultiFramedRTPSource {
public:
  static RawVideoRTPSource* createNew(UsageEnvironment& env, Groupsock* RTPgs,
				      unsigned char rtpPayloadFormat,
				      unsigned rtpTimestampFrequency);

  u_int16_t currentLineNumber() const; // of the most recently-read/processed scan line
  u_int8_t currentLineFieldId() const; // of the most recently-read/processed scan line (0 or 1)
  u_int16_t currentOffsetWithinLine() const; // of the most recently-read/processed scan line

protected:
  RawVideoRTPSource(UsageEnvironment& env, Groupsock* RTPgs,
		    unsigned char rtpPayloadFormat,
                    unsigned rtpTimestampFrequency = 90000);
      // called only by createNew()

  virtual ~RawVideoRTPSource();

protected:
  // redefined virtual functions:
  virtual Boolean processSpecialHeader(BufferedPacket* packet,
                                       unsigned& resultSpecialHeaderSize);
  virtual char const* MIMEtype() const;

private:
  unsigned fNumLines; // in the most recently read packet
  unsigned fNextLine; // index of the next AU Header to read
  struct LineHeader* fLineHeaders;

  friend class RawVideoBufferedPacket;
};

#endif
