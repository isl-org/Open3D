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
// A filter that parses a DV input stream into DV frames to deliver to the downstream object
// C++ header

#ifndef _DV_VIDEO_STREAM_FRAMER_HH
#define _DV_VIDEO_STREAM_FRAMER_HH

#ifndef _FRAMED_FILTER_HH
#include "FramedFilter.hh"
#endif

#define DV_DIF_BLOCK_SIZE 80
#define DV_NUM_BLOCKS_PER_SEQUENCE 150
#define DV_SAVED_INITIAL_BLOCKS_SIZE ((DV_NUM_BLOCKS_PER_SEQUENCE+6-1)*DV_DIF_BLOCK_SIZE)
    /* enough data to ensure that it contains an intact 6-block header (which occurs at the start of a 150-block sequence) */

class DVVideoStreamFramer: public FramedFilter {
public:
  static DVVideoStreamFramer*
  createNew(UsageEnvironment& env, FramedSource* inputSource,
	    Boolean sourceIsSeekable = False, Boolean leavePresentationTimesUnmodified = False);
      // Set "sourceIsSeekable" to True if the input source is a seekable object (e.g. a file), and the server that uses us
      // does a seek-to-zero on the source before reading from it.  (Our RTSP server implementation does this.)
  char const* profileName();
  Boolean getFrameParameters(unsigned& frameSize/*bytes*/, double& frameDuration/*microseconds*/);

protected:
  DVVideoStreamFramer(UsageEnvironment& env, FramedSource* inputSource,
		      Boolean sourceIsSeekable, Boolean leavePresentationTimesUnmodified);
      // called only by createNew(), or by subclass constructors
  virtual ~DVVideoStreamFramer();

protected:
  // redefined virtual functions:
  virtual Boolean isDVVideoStreamFramer() const;
  virtual void doGetNextFrame();

protected:
  void getAndDeliverData(); // used to implement "doGetNextFrame()"
  static void afterGettingFrame(void* clientData, unsigned frameSize,
                                unsigned numTruncatedBytes,
                                struct timeval presentationTime,
                                unsigned durationInMicroseconds);
  void afterGettingFrame(unsigned frameSize, unsigned numTruncatedBytes, struct timeval presentationTime);
  void getProfile();

protected:
  Boolean fLeavePresentationTimesUnmodified;
  void const* fOurProfile;
  struct timeval fNextFramePresentationTime;
  unsigned char fSavedInitialBlocks[DV_SAVED_INITIAL_BLOCKS_SIZE];
  char fInitialBlocksPresent;
  Boolean fSourceIsSeekable;
};

#endif
