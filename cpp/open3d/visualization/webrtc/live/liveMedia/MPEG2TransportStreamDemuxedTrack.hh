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
// A media track, demultiplexed from a MPEG Transport Stream file
// C++ header

#ifndef _MPEG2_TRANSPORT_STREAM_DEMUXED_TRACK_HH
#define _MPEG2_TRANSPORT_STREAM_DEMUXED_TRACK_HH

#ifndef _MPEG2_TRANSPORT_STREAM_DEMUX_HH
#include "MPEG2TransportStreamDemux.hh"
#endif

class MPEG2TransportStreamDemuxedTrack: public FramedSource {
public:
  MPEG2TransportStreamDemuxedTrack(class MPEG2TransportStreamParser& ourParser, u_int16_t pid);
  virtual ~MPEG2TransportStreamDemuxedTrack();

private:
  // redefined virtual functions:
  virtual void doGetNextFrame();

private: // We are accessed only by "MPEG2TransportStreamParser" (a friend)
  friend class MPEG2TransportStreamParser;
  unsigned char* to() { return fTo; }
  unsigned maxSize() { return fMaxSize; }
  unsigned& frameSize() { return fFrameSize; }
  unsigned& numTruncatedBytes() { return fNumTruncatedBytes; }
  struct timeval& presentationTime() { return fPresentationTime; }

private:
  class MPEG2TransportStreamParser& fOurParser;
  u_int16_t fPID;
};

#endif
