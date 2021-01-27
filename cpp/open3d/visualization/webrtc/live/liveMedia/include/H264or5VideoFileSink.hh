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
// H.264 or H.265 Video File Sinks
// C++ header

#ifndef _H264_OR_5_VIDEO_FILE_SINK_HH
#define _H264_OR_5_VIDEO_FILE_SINK_HH

#ifndef _FILE_SINK_HH
#include "FileSink.hh"
#endif

class H264or5VideoFileSink: public FileSink {
protected:
  H264or5VideoFileSink(UsageEnvironment& env, FILE* fid,
		       unsigned bufferSize, char const* perFrameFileNamePrefix,
		       char const* sPropParameterSetsStr1,
		       char const* sPropParameterSetsStr2 = NULL,
		       char const* sPropParameterSetsStr3 = NULL);
      // we're an abstract base class
  virtual ~H264or5VideoFileSink();

protected: // redefined virtual functions:
  virtual void afterGettingFrame(unsigned frameSize, unsigned numTruncatedBytes, struct timeval presentationTime);

private:
  char const* fSPropParameterSetsStr[3];
  Boolean fHaveWrittenFirstFrame;
};

#endif
