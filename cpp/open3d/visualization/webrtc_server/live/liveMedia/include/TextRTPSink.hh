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
// A generic RTP sink for text codecs (abstract base class)
// C++ header

#ifndef _TEXT_RTP_SINK_HH
#define _TEXT_RTP_SINK_HH

#ifndef _MULTI_FRAMED_RTP_SINK_HH
#include "MultiFramedRTPSink.hh"
#endif

class TextRTPSink: public MultiFramedRTPSink {
protected:
  TextRTPSink(UsageEnvironment& env,
	      Groupsock* rtpgs, unsigned char rtpPayloadType,
	      unsigned rtpTimestampFrequency,
	      char const* rtpPayloadFormatName);
  // (we're an abstract base class)
  virtual ~TextRTPSink();

private: // redefined virtual functions:
  virtual char const* sdpMediaType() const;
};

#endif
