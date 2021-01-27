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
// A media sink that takes - as input - a MPEG Transport Stream, and outputs a series
// of MPEG Transport Stream files, each representing a segment of the input stream,
// suitable for HLS (Apple's "HTTP Live Streaming").
// C++ header

#ifndef _HLS_SEGMENTER_HH
#define _HLS_SEGMENTER_HH

#ifndef _MEDIA_SINK_HH
#include "MediaSink.hh"
#endif

class HLSSegmenter: public MediaSink {
public:
  typedef void (onEndOfSegmentFunc)(void* clientData,
				    char const* segmentFileName, double segmentDuration);
  static HLSSegmenter* createNew(UsageEnvironment& env,
				 unsigned segmentationDuration, char const* fileNamePrefix,
				 onEndOfSegmentFunc* onEndOfSegmentFunc = NULL,
				 void* onEndOfSegmentClientData = NULL);

private:
  HLSSegmenter(UsageEnvironment& env, unsigned segmentationDuration, char const* fileNamePrefix,
	       onEndOfSegmentFunc* onEndOfSegmentFunc, void* onEndOfSegmentClientData);
    // called only by createNew()
  virtual ~HLSSegmenter();

  static void ourEndOfSegmentHandler(void* clientData, double segmentDuration);
  void ourEndOfSegmentHandler(double segmentDuration);

  Boolean openNextOutputSegment();

  static void afterGettingFrame(void* clientData, unsigned frameSize,
                                unsigned numTruncatedBytes,
                                struct timeval presentationTime,
                                unsigned durationInMicroseconds);
  virtual void afterGettingFrame(unsigned frameSize,
                                 unsigned numTruncatedBytes);

  static void ourOnSourceClosure(void* clientData);
  void ourOnSourceClosure();

private: // redefined virtual functions:
  virtual Boolean sourceIsCompatibleWithUs(MediaSource& source);
  virtual Boolean continuePlaying();

private:
  unsigned fSegmentationDuration;
  char const* fFileNamePrefix;
  onEndOfSegmentFunc* fOnEndOfSegmentFunc;
  void* fOnEndOfSegmentClientData;
  Boolean fHaveConfiguredUpstreamSource;
  unsigned fCurrentSegmentCounter;
  char* fOutputSegmentFileName;
  FILE* fOutFid;
  unsigned char* fOutputFileBuffer;
};

#endif
