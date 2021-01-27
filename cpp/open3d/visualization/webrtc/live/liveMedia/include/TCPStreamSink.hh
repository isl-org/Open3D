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
// A sink representing a TCP output stream
// C++ header

#ifndef _TCP_STREAM_SINK_HH
#define _TCP_STREAM_SINK_HH

#ifndef _MEDIA_SINK_HH
#include "MediaSink.hh"
#endif

#define TCP_STREAM_SINK_BUFFER_SIZE 10000

class TCPStreamSink: public MediaSink {
public:
  static TCPStreamSink* createNew(UsageEnvironment& env, int socketNum);
  // "socketNum" is the socket number of an existing, writable TCP socket (which should be non-blocking).
  // The caller is responsible for closing this socket later (when this object no longer exists).

protected:
  TCPStreamSink(UsageEnvironment& env, int socketNum); // called only by "createNew()"
  virtual ~TCPStreamSink();

protected:
  // Redefined virtual functions:
  virtual Boolean continuePlaying();

private:
  void processBuffer(); // common routine, called from both the 'socket writable' and 'incoming data' handlers below

  static void socketWritableHandler(void* clientData, int mask);
  void socketWritableHandler1();

  static void afterGettingFrame(void* clientData, unsigned frameSize, unsigned numTruncatedBytes,
				struct timeval /*presentationTime*/, unsigned /*durationInMicroseconds*/);
  void afterGettingFrame(unsigned frameSize, unsigned numTruncatedBytes);

  static void ourOnSourceClosure(void* clientData);
  void ourOnSourceClosure1();

  unsigned numUnwrittenBytes() const { return fUnwrittenBytesEnd - fUnwrittenBytesStart; }
  unsigned freeBufferSpace() const { return TCP_STREAM_SINK_BUFFER_SIZE - fUnwrittenBytesEnd; }

private:
  unsigned char fBuffer[TCP_STREAM_SINK_BUFFER_SIZE];
  unsigned fUnwrittenBytesStart, fUnwrittenBytesEnd;
  Boolean fInputSourceIsOpen, fOutputSocketIsWritable;
  int fOutputSocketNum;
};

#endif
