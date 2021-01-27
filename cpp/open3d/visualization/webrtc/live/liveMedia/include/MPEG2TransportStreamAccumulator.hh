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
// Collects a stream of incoming MPEG Transport Stream packets into
//  a chunk sufficiently large to send in a single outgoing (RTP or UDP) packet.
// C++ header

#ifndef _MPEG2_TRANSPORT_STREAM_ACCUMULATOR_HH
#define _MPEG_TRANSPORT_STREAM_ACCUMULATOR_HH

#ifndef _FRAMED_FILTER_HH
#include "FramedFilter.hh"
#endif

class MPEG2TransportStreamAccumulator: public FramedFilter {
public:
  static MPEG2TransportStreamAccumulator* createNew(UsageEnvironment& env,
						    FramedSource* inputSource,
						    unsigned maxPacketSize = 1456);

protected:
  MPEG2TransportStreamAccumulator(UsageEnvironment& env,
				  FramedSource* inputSource, unsigned maxPacketSize);
      // called only by createNew()
  virtual ~MPEG2TransportStreamAccumulator();

private:
  // redefined virtual functions:
  virtual void doGetNextFrame();

private:
  static void afterGettingFrame(void* clientData, unsigned frameSize,
                                unsigned numTruncatedBytes,
                                struct timeval presentationTime,
                                unsigned durationInMicroseconds);
  void afterGettingFrame1(unsigned frameSize,
                          unsigned numTruncatedBytes,
                          struct timeval presentationTime,
                          unsigned durationInMicroseconds);

private:
  unsigned const fDesiredPacketSize;
  unsigned fNumBytesGathered;
};

#endif

#ifndef _MP3_TRANSCODER_HH
#define _MP3_TRANSCODER_HH

#ifndef _MP3_ADU_HH
#include "MP3ADU.hh"
#endif
#ifndef _MP3_ADU_TRANSCODER_HH
#include "MP3ADUTranscoder.hh"
#endif

class MP3Transcoder: public MP3FromADUSource {
public:
  static MP3Transcoder* createNew(UsageEnvironment& env,
				  unsigned outBitrate /* in kbps */,
				  FramedSource* inputSource);

protected:
  MP3Transcoder(UsageEnvironment& env,
		MP3ADUTranscoder* aduTranscoder);
      // called only by createNew()
  virtual ~MP3Transcoder();
};

#endif
