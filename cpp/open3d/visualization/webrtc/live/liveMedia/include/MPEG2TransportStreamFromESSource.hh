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
// A filter for converting one or more MPEG Elementary Streams
// to a MPEG-2 Transport Stream
// C++ header

#ifndef _MPEG2_TRANSPORT_STREAM_FROM_ES_SOURCE_HH
#define _MPEG2_TRANSPORT_STREAM_FROM_ES_SOURCE_HH

#ifndef _MPEG2_TRANSPORT_STREAM_MULTIPLEXOR_HH
#include "MPEG2TransportStreamMultiplexor.hh"
#endif

class MPEG2TransportStreamFromESSource: public MPEG2TransportStreamMultiplexor {
public:
  static MPEG2TransportStreamFromESSource* createNew(UsageEnvironment& env);

  void addNewVideoSource(FramedSource* inputSource, int mpegVersion, int16_t PID = -1);
      // Note: For MPEG-4 video, set "mpegVersion" to 4; for H.264 video, set "mpegVersion" to 5;
      //     for H.265 video, set "mpegVersion" to 6
  void addNewAudioSource(FramedSource* inputSource, int mpegVersion, int16_t PID = -1);
      // Note: For Opus audio, set "mpegVersion" to 3
  
      // Note: In these functions, if "PID" is not -1, then it (currently, just the low 8 bits)
      // is used as the stream's PID.  Otherwise (if "PID" is -1) the 'stream_id' is used as
      // the PID.

  static unsigned maxInputESFrameSize;

protected:
  MPEG2TransportStreamFromESSource(UsageEnvironment& env);
      // called only by createNew()
  virtual ~MPEG2TransportStreamFromESSource();

  void addNewInputSource(FramedSource* inputSource,
			 u_int8_t streamId, int mpegVersion, int16_t PID = -1);
  // used to implement addNew*Source() above

private:
  // Redefined virtual functions:
  virtual void doStopGettingFrames();
  virtual void awaitNewBuffer(unsigned char* oldBuffer);

private:
  friend class InputESSourceRecord;
  class InputESSourceRecord* fInputSources;
  unsigned fVideoSourceCounter, fAudioSourceCounter;
  Boolean fAwaitingBackgroundDelivery;
};

#endif
