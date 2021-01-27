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
// A parser for a Matroska file.
// C++ header

#ifndef _MATROSKA_FILE_PARSER_HH

#ifndef _STREAM_PARSER_HH
#include "StreamParser.hh"
#endif
#ifndef _MATROSKA_FILE_HH
#include "MatroskaFile.hh"
#endif
#ifndef _EBML_NUMBER_HH
#include "EBMLNumber.hh"
#endif

// An enum representing the current state of the parser:
enum MatroskaParseState {
  PARSING_START_OF_FILE,
  LOOKING_FOR_TRACKS,
  PARSING_TRACK,
  PARSING_CUES,
  LOOKING_FOR_CLUSTER,
  LOOKING_FOR_BLOCK,
  PARSING_BLOCK,
  DELIVERING_FRAME_WITHIN_BLOCK,
  DELIVERING_FRAME_BYTES
};

class MatroskaFileParser: public StreamParser {
public:
  MatroskaFileParser(MatroskaFile& ourFile, FramedSource* inputSource,
		     FramedSource::onCloseFunc* onEndFunc, void* onEndClientData,
		     MatroskaDemux* ourDemux = NULL);
  virtual ~MatroskaFileParser();

  void seekToTime(double& seekNPT);

  // StreamParser 'client continue' function:
  static void continueParsing(void* clientData, unsigned char* ptr, unsigned size, struct timeval presentationTime);
  void continueParsing();

private:
  // Parsing functions:
  Boolean parse();
    // returns True iff we have finished parsing to the end of all 'Track' headers (on initialization)

  Boolean parseStartOfFile();
  void lookForNextTrack();
  Boolean parseTrack();
  Boolean parseCues();

  void lookForNextBlock();
  void parseBlock();
  Boolean deliverFrameWithinBlock();
  void deliverFrameBytes();

  void getCommonFrameBytes(MatroskaTrack* track, u_int8_t* to, unsigned numBytesToGet, unsigned numBytesToSkip);

  Boolean parseEBMLNumber(EBMLNumber& num);
  Boolean parseEBMLIdAndSize(EBMLId& id, EBMLDataSize& size);
  Boolean parseEBMLVal_unsigned64(EBMLDataSize& size, u_int64_t& result);
  Boolean parseEBMLVal_unsigned(EBMLDataSize& size, unsigned& result);
  Boolean parseEBMLVal_float(EBMLDataSize& size, float& result);
  Boolean parseEBMLVal_string(EBMLDataSize& size, char*& result);
    // Note: "result" is dynamically allocated; the caller must delete[] it later
  Boolean parseEBMLVal_binary(EBMLDataSize& size, u_int8_t*& result);
    // Note: "result" is dynamically allocated; the caller must delete[] it later
  void skipHeader(EBMLDataSize const& size);
  void skipRemainingHeaderBytes(Boolean isContinuation);

  void setParseState();

  void seekToFilePosition(u_int64_t offsetInFile);
  void seekToEndOfFile();
  void resetStateAfterSeeking(); // common code, called by both of the above

private: // redefined virtual functions
  virtual void restoreSavedParserState();

private:
  // General state for parsing:
  MatroskaFile& fOurFile;
  FramedSource* fInputSource;
  FramedSource::onCloseFunc* fOnEndFunc;
  void* fOnEndClientData;
  MatroskaDemux* fOurDemux;
  MatroskaParseState fCurrentParseState;
  u_int64_t fCurOffsetInFile, fSavedCurOffsetInFile, fLimitOffsetInFile;

  // For skipping over (possibly large) headers:
  u_int64_t fNumHeaderBytesToSkip;

  // For parsing 'Seek ID's:
  EBMLId fLastSeekId;

  // Parameters of the most recently-parsed 'Cluster':
  unsigned fClusterTimecode;

  // Parameters of the most recently-parsed 'Block':
  unsigned fBlockSize;
  unsigned fBlockTrackNumber;
  short fBlockTimecode;
  unsigned fNumFramesInBlock;
  unsigned* fFrameSizesWithinBlock;

  // Parameters of the most recently-parsed frame within a 'Block':
  double fPresentationTimeOffset;
  unsigned fNextFrameNumberToDeliver;
  unsigned fCurOffsetWithinFrame, fSavedCurOffsetWithinFrame; // used if track->haveSubframes()

  // Parameters of the (sub)frame that's currently being delivered:
  u_int8_t* fCurFrameTo;
  unsigned fCurFrameNumBytesToGet;
  unsigned fCurFrameNumBytesToSkip;
};

#endif
