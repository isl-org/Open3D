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
// A parser for a MPEG Transport Stream
// C++ header

#ifndef _MPEG2_TRANSPORT_STREAM_PARSER_HH

#ifndef _STREAM_PARSER_HH
#include "StreamParser.hh"
#endif
#ifndef _MPEG2_TRANSPORT_STREAM_DEMUXED_TRACK_HH
#include "MPEG2TransportStreamDemuxedTrack.hh"
#endif
#ifndef _MEDIA_SINK_HH
#include "MediaSink.hh"
#endif

// A descriptor that describes the state of each known PID:
enum PIDType { PAT, PMT, STREAM };

class PIDState {
protected: // we're a virtual base class
  PIDState(MPEG2TransportStreamParser& parser, u_int16_t pid, PIDType pidType);
public:
  virtual ~PIDState();

public:
  MPEG2TransportStreamParser& ourParser;
  u_int16_t PID;
  PIDType type;
};

class PIDState_PAT : public PIDState {
public:
  PIDState_PAT(MPEG2TransportStreamParser& parser, u_int16_t pid);
protected:
  virtual ~PIDState_PAT();
};

class PIDState_PMT : public PIDState {
public:
  PIDState_PMT(MPEG2TransportStreamParser& parser, u_int16_t pid, u_int16_t programNumber);
protected:
  virtual ~PIDState_PMT();

public:
  u_int16_t program_number;
};

class PIDState_STREAM : public PIDState {
public:
  PIDState_STREAM(MPEG2TransportStreamParser& parser, u_int16_t pid, u_int16_t programNumber, u_int8_t streamType);
protected:
  virtual ~PIDState_STREAM();

public:
  u_int16_t program_number;
  u_int8_t stream_type;
  double lastSeenPTS;
  MPEG2TransportStreamDemuxedTrack* streamSource;
  MediaSink* streamSink;
};


// Descriptions of known "stream_type"s:
class StreamType {
public:
  char const* description;
  enum dataType { AUDIO, VIDEO, DATA, TEXT, UNKNOWN } dataType;
  char const* filenameSuffix;

public:
  StreamType(char const* description = "unknown", enum dataType dataType = UNKNOWN,
	     char const* filenameSuffix = "");
};


class MPEG2TransportStreamParser: public StreamParser {
public:
  MPEG2TransportStreamParser(FramedSource* inputSource,
			     FramedSource::onCloseFunc* onEndFunc, void* onEndClientData);
  virtual ~MPEG2TransportStreamParser();

  UsageEnvironment& envir();

  // StreamParser 'client continue' function:
  static void continueParsing(void* clientData, unsigned char* ptr, unsigned size, struct timeval presentationTime);
  void continueParsing();

private:
  // Parsing functions:
  friend class MPEG2TransportStreamDemuxedTrack;
  Boolean parse(); // returns True iff we have finished parsing all BOS pages (on initialization)

  u_int8_t parseAdaptationField();
  Boolean processDataBytes(u_int16_t PID, Boolean pusi, unsigned numDataBytes);

  void parsePAT(Boolean pusi, unsigned numDataBytes);
  void parsePMT(PIDState_PMT* pidState, Boolean pusi, unsigned numDataBytes);
  void parseStreamDescriptors(unsigned numDescriptorBytes);
  Boolean processStreamPacket(PIDState_STREAM* pidState, Boolean pusi, unsigned numDataBytes);
  unsigned parsePESHeader(PIDState_STREAM* pidState, unsigned numDataBytes);

private: // redefined virtual functions
  virtual void restoreSavedParserState();

private:
  // General state for parsing:
  FramedSource* fInputSource;
  Boolean fAmCurrentlyParsing;
  FramedSource::onCloseFunc* fOnEndFunc;
  void* fOnEndClientData;
  PIDState** fPIDState;
  double fLastSeenPCR;
};

#endif
