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
// A media track, demultiplexed from an Ogg file
// C++ header

#ifndef _OGG_DEMUXED_TRACK_HH
#define _OGG_DEMUXED_TRACK_HH

#ifndef _FRAMED_SOURCE_HH
#include "FramedSource.hh"
#endif

class OggDemux; // forward

class OggDemuxedTrack: public FramedSource {
private: // We are created only by a OggDemux (a friend)
  friend class OggDemux;
  OggDemuxedTrack(UsageEnvironment& env, unsigned trackNumber, OggDemux& sourceDemux);
  virtual ~OggDemuxedTrack();

private:
  // redefined virtual functions:
  virtual void doGetNextFrame();
  virtual char const* MIMEtype() const;

private: // We are accessed only by OggDemux and by OggFileParser (a friend)
  friend class OggFileParser;
  unsigned char*& to() { return fTo; }
  unsigned& maxSize() { return fMaxSize; }
  unsigned& frameSize() { return fFrameSize; }
  unsigned& numTruncatedBytes() { return fNumTruncatedBytes; }
  struct timeval& presentationTime() { return fPresentationTime; }
  unsigned& durationInMicroseconds() { return fDurationInMicroseconds; }
  struct timeval& nextPresentationTime() { return fNextPresentationTime; }

private:
  unsigned fOurTrackNumber;
  OggDemux& fOurSourceDemux;
  Boolean fCurrentPageIsContinuation;
  struct timeval fNextPresentationTime;
};

#endif
