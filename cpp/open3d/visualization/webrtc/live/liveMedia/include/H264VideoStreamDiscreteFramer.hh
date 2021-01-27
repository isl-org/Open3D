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
// A simplified version of "H264VideoStreamFramer" that takes only complete,
// discrete frames (rather than an arbitrary byte stream) as input.
// This avoids the parsing and data copying overhead of the full
// "H264VideoStreamFramer".
// C++ header

#ifndef _H264_VIDEO_STREAM_DISCRETE_FRAMER_HH
#define _H264_VIDEO_STREAM_DISCRETE_FRAMER_HH

#ifndef _H264_OR_5_VIDEO_STREAM_DISCRETE_FRAMER_HH
#include "H264or5VideoStreamDiscreteFramer.hh"
#endif

class H264VideoStreamDiscreteFramer: public H264or5VideoStreamDiscreteFramer {
public:
  static H264VideoStreamDiscreteFramer*
  createNew(UsageEnvironment& env, FramedSource* inputSource,
	    Boolean includeStartCodeInOutput = False, Boolean insertAccessUnitDelimiters = False);

protected:
  H264VideoStreamDiscreteFramer(UsageEnvironment& env, FramedSource* inputSource,
				Boolean includeStartCodeInOutput, Boolean insertAccessUnitDelimiters);
      // called only by createNew()
  virtual ~H264VideoStreamDiscreteFramer();

private:
  // redefined virtual functions:
  virtual Boolean isH264VideoStreamFramer() const;
};

#endif
