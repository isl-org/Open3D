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
// A media track, demultiplexed from a Matroska file
// C++ header

#ifndef _MATROSKA_DEMUXED_TRACK_HH
#define _MATROSKA_DEMUXED_TRACK_HH

#ifndef _FRAMED_SOURCE_HH
#include "FramedSource.hh"
#endif

class MatroskaDemux; // forward

class MatroskaDemuxedTrack: public FramedSource {
public:
  void seekToTime(double& seekNPT);

private: // We are created only by a MatroskaDemux (a friend)
  friend class MatroskaDemux;
  MatroskaDemuxedTrack(UsageEnvironment& env, unsigned trackNumber, MatroskaDemux& sourceDemux);
  virtual ~MatroskaDemuxedTrack();

private:
  // redefined virtual functions:
  virtual void doGetNextFrame();
  virtual char const* MIMEtype() const;

private: // We are accessed only by MatroskaDemux and by MatroskaFileParser (a friend)
  friend class MatroskaFileParser;
  unsigned char* to() { return fTo; }
  unsigned maxSize() { return fMaxSize; }
  unsigned& frameSize() { return fFrameSize; }
  unsigned& numTruncatedBytes() { return fNumTruncatedBytes; }
  struct timeval& presentationTime() { return fPresentationTime; }
  unsigned& durationInMicroseconds() { return fDurationInMicroseconds; }

  struct timeval& prevPresentationTime() { return fPrevPresentationTime; }
  int& durationImbalance() { return fDurationImbalance; }

private:
  unsigned fOurTrackNumber;
  MatroskaDemux& fOurSourceDemux;
  struct timeval fPrevPresentationTime;
  int fDurationImbalance;
  unsigned fOpusTrackNumber; // hack for Opus audio
};

#endif
