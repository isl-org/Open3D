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
// RTP sink for T.140 text (RFC 2793)
// C++ header

#ifndef _T140_TEXT_RTP_SINK_HH
#define _T140_TEXT_RTP_SINK_HH

#ifndef _TEXT_RTP_SINK_HH
#include "TextRTPSink.hh"
#endif
#ifndef _FRAMED_FILTER_HH
#include "FramedFilter.hh"
#endif

class T140IdleFilter;

class T140TextRTPSink: public TextRTPSink {
public:
  static T140TextRTPSink* createNew(UsageEnvironment& env, Groupsock* RTPgs, unsigned char rtpPayloadFormat);

protected:
  T140TextRTPSink(UsageEnvironment& env, Groupsock* RTPgs, unsigned char rtpPayloadFormat);
	// called only by createNew()

  virtual ~T140TextRTPSink();

protected: // redefined virtual functions:
  virtual Boolean continuePlaying();
  virtual void doSpecialFrameHandling(unsigned fragmentationOffset,
                                      unsigned char* frameStart,
                                      unsigned numBytesInFrame,
                                      struct timeval framePresentationTime,
                                      unsigned numRemainingBytes);
  virtual Boolean frameCanAppearAfterPacketStart(unsigned char const* frameStart,
						 unsigned numBytesInFrame) const;

protected:
  T140IdleFilter* fOurIdleFilter;
  Boolean fAreInIdlePeriod;
};


////////// T140IdleFilter definition //////////

// Because the T.140 text RTP payload format specification recommends that (empty) RTP packets be sent during 'idle periods'
// when no new text is available, we implement "T140TextRTPSink" using a separate "T140IdleFilter" class - sitting in front
// -  that delivers, to the "T140TextRTPSink", a continuous sequence of (possibly) empty frames.
// (Note: This class should be used only by "T140TextRTPSink", or a subclass.)

class T140IdleFilter: public FramedFilter {
public:
  T140IdleFilter(UsageEnvironment& env, FramedSource* inputSource);
  virtual ~T140IdleFilter();

private: // redefined virtual functions:
  virtual void doGetNextFrame();
  virtual void doStopGettingFrames();

private:
  static void afterGettingFrame(void* clientData, unsigned frameSize,
				unsigned numTruncatedBytes,
                                struct timeval presentationTime,
                                unsigned durationInMicroseconds);
  void afterGettingFrame(unsigned frameSize,
			 unsigned numTruncatedBytes,
			 struct timeval presentationTime,
			 unsigned durationInMicroseconds);

  static void handleIdleTimeout(void* clientData);
  void handleIdleTimeout();

  void deliverFromBuffer();
  void deliverEmptyFrame();

  static void onSourceClosure(void* clientData);
  void onSourceClosure();

private:
  TaskToken fIdleTimerTask;
  unsigned fBufferSize, fNumBufferedBytes;
  char* fBuffer;
  unsigned fBufferedNumTruncatedBytes; // a count of truncated bytes from the upstream
  struct timeval fBufferedDataPresentationTime;
  unsigned fBufferedDataDurationInMicroseconds;
};

#endif
