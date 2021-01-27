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
// A parser for an Ogg file
// C++ header

#ifndef _OGG_FILE_PARSER_HH

#ifndef _STREAM_PARSER_HH
#include "StreamParser.hh"
#endif
#ifndef _OGG_FILE_HH
#include "OggFile.hh"
#endif

// An enum representing the current state of the parser:
enum OggParseState {
  PARSING_START_OF_FILE,
  PARSING_AND_DELIVERING_PAGES,
  DELIVERING_PACKET_WITHIN_PAGE
};

// A structure that counts the sizes of 'packets' given by each page's "segment_table":
class PacketSizeTable {
public:
  PacketSizeTable(unsigned number_page_segments);
  ~PacketSizeTable();

  unsigned numCompletedPackets; // will be <= "number_page_segments"
  unsigned* size; // an array of sizes of each of the packets
  unsigned totSizes;
  unsigned nextPacketNumToDeliver;
  Boolean lastPacketIsIncomplete; // iff the last segment's 'lacing' was 255
};

class OggFileParser: public StreamParser {
public:
  OggFileParser(OggFile& ourFile, FramedSource* inputSource,
		FramedSource::onCloseFunc* onEndFunc, void* onEndClientData,
		OggDemux* ourDemux = NULL);
  virtual ~OggFileParser();

  // StreamParser 'client continue' function:
  static void continueParsing(void* clientData, unsigned char* ptr, unsigned size, struct timeval presentationTime);
  void continueParsing();

private:
  Boolean needHeaders() { return fNumUnfulfilledTracks > 0; }

  // Parsing functions:
  Boolean parse(); // returns True iff we have finished parsing all BOS pages (on initialization)

  Boolean parseStartOfFile();
  u_int8_t parseInitialPage(); // returns the 'header_type_flag' byte
  void parseAndDeliverPages();
  Boolean parseAndDeliverPage();
  Boolean deliverPacketWithinPage();
  void parseStartOfPage(u_int8_t& header_type_flag, u_int32_t& bitstream_serial_number);

  Boolean validateHeader(OggTrack* track, u_int8_t const* p, unsigned headerSize);

private:
  // General state for parsing:
  OggFile& fOurFile;
  FramedSource* fInputSource;
  FramedSource::onCloseFunc* fOnEndFunc;
  void* fOnEndClientData;
  OggDemux* fOurDemux;
  OggParseState fCurrentParseState;

  unsigned fNumUnfulfilledTracks;
  PacketSizeTable* fPacketSizeTable;
  u_int32_t fCurrentTrackNumber;
  u_int8_t* fSavedPacket; // used to temporarily save a copy of a 'packet' from a page
};

#endif
