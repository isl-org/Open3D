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
// A filter that breaks up an AC3 audio elementary stream into frames
// C++ header

#ifndef _AC3_AUDIO_STREAM_FRAMER_HH
#define _AC3_AUDIO_STREAM_FRAMER_HH

#ifndef _FRAMED_FILTER_HH
#include "FramedFilter.hh"
#endif

class AC3AudioStreamFramer: public FramedFilter {
public:
  static AC3AudioStreamFramer*
  createNew(UsageEnvironment& env, FramedSource* inputSource,
	    unsigned char streamCode = 0);
  // If "streamCode" != 0, then we assume that there's a 1-byte code at the beginning of each chunk of data that we read from
  // our source.  If that code is not the value we want, we discard the chunk of data.
  // However, if "streamCode" == 0 (the default), then we don't expect this 1-byte code.

  unsigned samplingRate();

  void flushInput(); // called if there is a discontinuity (seeking) in the input

private:
  AC3AudioStreamFramer(UsageEnvironment& env, FramedSource* inputSource,
		       unsigned char streamCode);
      // called only by createNew()
  virtual ~AC3AudioStreamFramer();

  static void handleNewData(void* clientData,
			    unsigned char* ptr, unsigned size,
			    struct timeval presentationTime);
  void handleNewData(unsigned char* ptr, unsigned size);

  void parseNextFrame();

private:
  // redefined virtual functions:
  virtual void doGetNextFrame();

private:
  struct timeval currentFramePlayTime() const;

private:
  struct timeval fNextFramePresentationTime;

private: // parsing state
  class AC3AudioStreamParser* fParser;
  unsigned char fOurStreamCode;
  friend class AC3AudioStreamParser; // hack
};

#endif
