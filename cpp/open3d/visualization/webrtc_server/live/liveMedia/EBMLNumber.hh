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
// EBML numbers (ids and sizes)
// C++ header

#ifndef _EBML_NUMBER_HH
#define _EBML_NUMBER_HH

#include "NetCommon.h"
#include "Boolean.hh"
#include <stdio.h>

#define EBML_NUMBER_MAX_LEN 8

class EBMLNumber {
public:
  EBMLNumber(Boolean stripLeading1 = True);
  virtual ~EBMLNumber();

  u_int64_t val() const;  
  char* hexString() const; // used for debugging
  Boolean operator==(u_int64_t arg2) const { return val() == arg2; }
  Boolean operator!=(u_int64_t arg2) const { return !(*this == arg2); }

public:
  Boolean stripLeading1;
  unsigned len;
  u_int8_t data[EBML_NUMBER_MAX_LEN];
};

// Definitions of some Matroska/EBML IDs (including the ones that we check for):
#define MATROSKA_ID_EBML 0x1A45DFA3
#define MATROSKA_ID_VOID 0xEC
#define MATROSKA_ID_CRC_32 0xBF
#define MATROSKA_ID_SEGMENT 0x18538067
#define MATROSKA_ID_SEEK_HEAD 0x114D9B74
#define MATROSKA_ID_SEEK 0x4DBB
#define MATROSKA_ID_SEEK_ID 0x53AB
#define MATROSKA_ID_SEEK_POSITION 0x53AC
#define MATROSKA_ID_INFO 0x1549A966
#define MATROSKA_ID_SEGMENT_UID 0x73A4
#define MATROSKA_ID_TIMECODE_SCALE 0x2AD7B1
#define MATROSKA_ID_DURATION 0x4489
#define MATROSKA_ID_DATE_UTC 0x4461
#define MATROSKA_ID_TITLE 0x7BA9
#define MATROSKA_ID_MUXING_APP 0x4D80
#define MATROSKA_ID_WRITING_APP 0x5741
#define MATROSKA_ID_CLUSTER 0x1F43B675
#define MATROSKA_ID_TIMECODE 0xE7
#define MATROSKA_ID_POSITION 0xA7
#define MATROSKA_ID_PREV_SIZE 0xAB
#define MATROSKA_ID_SIMPLEBLOCK 0xA3
#define MATROSKA_ID_BLOCK_GROUP 0xA0
#define MATROSKA_ID_BLOCK 0xA1
#define MATROSKA_ID_BLOCK_DURATION 0x9B
#define MATROSKA_ID_REFERENCE_BLOCK 0xFB
#define MATROSKA_ID_TRACKS 0x1654AE6B
#define MATROSKA_ID_TRACK_ENTRY 0xAE
#define MATROSKA_ID_TRACK_NUMBER 0xD7
#define MATROSKA_ID_TRACK_UID 0x73C5
#define MATROSKA_ID_TRACK_TYPE 0x83
#define MATROSKA_ID_FLAG_ENABLED 0xB9
#define MATROSKA_ID_FLAG_DEFAULT 0x88
#define MATROSKA_ID_FLAG_FORCED 0x55AA
#define MATROSKA_ID_FLAG_LACING 0x9C
#define MATROSKA_ID_MIN_CACHE 0x6DE7
#define MATROSKA_ID_DEFAULT_DURATION 0x23E383
#define MATROSKA_ID_TRACK_TIMECODE_SCALE 0x23314F
#define MATROSKA_ID_MAX_BLOCK_ADDITION_ID 0x55EE
#define MATROSKA_ID_NAME 0x536E
#define MATROSKA_ID_LANGUAGE 0x22B59C
#define MATROSKA_ID_CODEC 0x86
#define MATROSKA_ID_CODEC_PRIVATE 0x63A2
#define MATROSKA_ID_CODEC_NAME 0x258688
#define MATROSKA_ID_CODEC_DECODE_ALL 0xAA
#define MATROSKA_ID_VIDEO 0xE0
#define MATROSKA_ID_FLAG_INTERLACED 0x9A
#define MATROSKA_ID_PIXEL_WIDTH 0xB0
#define MATROSKA_ID_PIXEL_HEIGHT 0xBA
#define MATROSKA_ID_DISPLAY_WIDTH 0x54B0
#define MATROSKA_ID_DISPLAY_HEIGHT 0x54BA
#define MATROSKA_ID_DISPLAY_UNIT 0x54B2
#define MATROSKA_ID_AUDIO 0xE1
#define MATROSKA_ID_SAMPLING_FREQUENCY 0xB5
#define MATROSKA_ID_OUTPUT_SAMPLING_FREQUENCY 0x78B5
#define MATROSKA_ID_CHANNELS 0x9F
#define MATROSKA_ID_BIT_DEPTH 0x6264
#define MATROSKA_ID_CONTENT_ENCODINGS 0x6D80
#define MATROSKA_ID_CONTENT_ENCODING 0x6240
#define MATROSKA_ID_CONTENT_COMPRESSION 0x5034
#define MATROSKA_ID_CONTENT_COMP_ALGO 0x4254
#define MATROSKA_ID_CONTENT_COMP_SETTINGS 0x4255
#define MATROSKA_ID_CONTENT_ENCRYPTION 0x5035
#define MATROSKA_ID_ATTACHMENTS 0x1941A469
#define MATROSKA_ID_ATTACHED_FILE 0x61A7
#define MATROSKA_ID_FILE_DESCRIPTION 0x467E
#define MATROSKA_ID_FILE_NAME 0x466E
#define MATROSKA_ID_FILE_MIME_TYPE 0x4660
#define MATROSKA_ID_FILE_DATA 0x465C
#define MATROSKA_ID_FILE_UID 0x46AE
#define MATROSKA_ID_CUES 0x1C53BB6B
#define MATROSKA_ID_CUE_POINT 0xBB
#define MATROSKA_ID_CUE_TIME 0xB3
#define MATROSKA_ID_CUE_TRACK_POSITIONS 0xB7
#define MATROSKA_ID_CUE_TRACK 0xF7
#define MATROSKA_ID_CUE_CLUSTER_POSITION 0xF1
#define MATROSKA_ID_CUE_BLOCK_NUMBER 0x5378
#define MATROSKA_ID_TAGS 0x1254C367
#define MATROSKA_ID_SEEK_PRE_ROLL 0x56BB
#define MATROSKA_ID_CODEC_DELAY 0x56AA
#define MATROSKA_ID_DISCARD_PADDING 0x75A2
#define MATROSKA_ID_COLOR_SPACE 0x2EB524
#define MATROSKA_ID_PRIMARIES 0x55BB

class EBMLId: public EBMLNumber {
public:
  EBMLId();
  virtual ~EBMLId();

  char const* stringName() const; // used for debugging
};

class EBMLDataSize: public EBMLNumber {
public:
  EBMLDataSize();
  virtual ~EBMLDataSize();
};

#endif
