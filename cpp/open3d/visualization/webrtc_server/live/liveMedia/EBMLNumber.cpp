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
// Implementation

#include "EBMLNumber.hh"

EBMLNumber::EBMLNumber(Boolean stripLeading1)
    : stripLeading1(stripLeading1), len(0) {}

EBMLNumber::~EBMLNumber() {}

char* EBMLNumber::hexString() const {
    // Originally +1, now +5 to suppress warnings.
    static char printBuf[2 * EBML_NUMBER_MAX_LEN + 5];

    char* to = printBuf;
    for (unsigned i = 0; i < len; ++i) {
        sprintf(to, "%02X", data[i]);
        to += 2;
    }

    return printBuf;
}

u_int64_t EBMLNumber::val() const {
    u_int64_t result = 0;

    for (unsigned i = 0; i < len; ++i) {
        result = result * 256 + data[i];
    }

    return result;
}

EBMLId::EBMLId() : EBMLNumber(False) {}

EBMLId::~EBMLId() {}

char const* EBMLId::stringName() const {
    switch (val()) {
        case MATROSKA_ID_EBML: {
            return "EBML";
        }
        case MATROSKA_ID_VOID: {
            return "Void";
        }
        case MATROSKA_ID_CRC_32: {
            return "CRC-32";
        }
        case MATROSKA_ID_SEGMENT: {
            return "Segment";
        }
        case MATROSKA_ID_SEEK_HEAD: {
            return "Seek Head";
        }
        case MATROSKA_ID_SEEK: {
            return "Seek";
        }
        case MATROSKA_ID_SEEK_ID: {
            return "Seek ID";
        }
        case MATROSKA_ID_SEEK_POSITION: {
            return "Seek Position";
        }
        case MATROSKA_ID_INFO: {
            return "Segment Info";
        }
        case MATROSKA_ID_SEGMENT_UID: {
            return "Segment UID";
        }
        case MATROSKA_ID_DURATION: {
            return "Segment Duration";
        }
        case MATROSKA_ID_TIMECODE_SCALE: {
            return "Timecode Scale";
        }
        case MATROSKA_ID_DATE_UTC: {
            return "Date (UTC)";
        }
        case MATROSKA_ID_TITLE: {
            return "Title";
        }
        case MATROSKA_ID_MUXING_APP: {
            return "Muxing App";
        }
        case MATROSKA_ID_WRITING_APP: {
            return "Writing App";
        }
        case MATROSKA_ID_CLUSTER: {
            return "Cluster";
        }
        case MATROSKA_ID_TIMECODE: {
            return "TimeCode";
        }
        case MATROSKA_ID_POSITION: {
            return "Position";
        }
        case MATROSKA_ID_PREV_SIZE: {
            return "Prev. Size";
        }
        case MATROSKA_ID_SIMPLEBLOCK: {
            return "SimpleBlock";
        }
        case MATROSKA_ID_BLOCK_GROUP: {
            return "Block Group";
        }
        case MATROSKA_ID_BLOCK: {
            return "Block";
        }
        case MATROSKA_ID_BLOCK_DURATION: {
            return "Block Duration";
        }
        case MATROSKA_ID_REFERENCE_BLOCK: {
            return "Reference Block";
        }
        case MATROSKA_ID_TRACKS: {
            return "Tracks";
        }
        case MATROSKA_ID_TRACK_ENTRY: {
            return "Track Entry";
        }
        case MATROSKA_ID_TRACK_NUMBER: {
            return "Track Number";
        }
        case MATROSKA_ID_TRACK_UID: {
            return "Track UID";
        }
        case MATROSKA_ID_TRACK_TYPE: {
            return "Track Type";
        }
        case MATROSKA_ID_FLAG_ENABLED: {
            return "Flag Enabled";
        }
        case MATROSKA_ID_FLAG_DEFAULT: {
            return "Flag Default";
        }
        case MATROSKA_ID_FLAG_FORCED: {
            return "Flag Forced";
        }
        case MATROSKA_ID_FLAG_LACING: {
            return "Flag Lacing";
        }
        case MATROSKA_ID_MIN_CACHE: {
            return "Min Cache";
        }
        case MATROSKA_ID_DEFAULT_DURATION: {
            return "Default Duration";
        }
        case MATROSKA_ID_TRACK_TIMECODE_SCALE: {
            return "Track Timecode Scale";
        }
        case MATROSKA_ID_MAX_BLOCK_ADDITION_ID: {
            return "Max Block Addition ID";
        }
        case MATROSKA_ID_NAME: {
            return "Name";
        }
        case MATROSKA_ID_LANGUAGE: {
            return "Language";
        }
        case MATROSKA_ID_CODEC: {
            return "Codec ID";
        }
        case MATROSKA_ID_CODEC_PRIVATE: {
            return "Codec Private";
        }
        case MATROSKA_ID_CODEC_NAME: {
            return "Codec Name";
        }
        case MATROSKA_ID_CODEC_DECODE_ALL: {
            return "Codec Decode All";
        }
        case MATROSKA_ID_VIDEO: {
            return "Video Settings";
        }
        case MATROSKA_ID_FLAG_INTERLACED: {
            return "Flag Interlaced";
        }
        case MATROSKA_ID_PIXEL_WIDTH: {
            return "Pixel Width";
        }
        case MATROSKA_ID_PIXEL_HEIGHT: {
            return "Pixel Height";
        }
        case MATROSKA_ID_DISPLAY_WIDTH: {
            return "Display Width";
        }
        case MATROSKA_ID_DISPLAY_HEIGHT: {
            return "Display Height";
        }
        case MATROSKA_ID_DISPLAY_UNIT: {
            return "Display Unit";
        }
        case MATROSKA_ID_AUDIO: {
            return "Audio Settings";
        }
        case MATROSKA_ID_SAMPLING_FREQUENCY: {
            return "Sampling Frequency";
        }
        case MATROSKA_ID_OUTPUT_SAMPLING_FREQUENCY: {
            return "Output Sampling Frequency";
        }
        case MATROSKA_ID_CHANNELS: {
            return "Channels";
        }
        case MATROSKA_ID_BIT_DEPTH: {
            return "Bit Depth";
        }
        case MATROSKA_ID_CONTENT_ENCODINGS: {
            return "Content Encodings";
        }
        case MATROSKA_ID_CONTENT_ENCODING: {
            return "Content Encoding";
        }
        case MATROSKA_ID_CONTENT_COMPRESSION: {
            return "Content Compression";
        }
        case MATROSKA_ID_CONTENT_COMP_ALGO: {
            return "Content Compression Algorithm";
        }
        case MATROSKA_ID_CONTENT_COMP_SETTINGS: {
            return "Content Compression Settings";
        }
        case MATROSKA_ID_CONTENT_ENCRYPTION: {
            return "Content Encryption";
        }
        case MATROSKA_ID_ATTACHMENTS: {
            return "Attachments";
        }
        case MATROSKA_ID_ATTACHED_FILE: {
            return "Attached File";
        }
        case MATROSKA_ID_FILE_DESCRIPTION: {
            return "File Description";
        }
        case MATROSKA_ID_FILE_NAME: {
            return "File Name";
        }
        case MATROSKA_ID_FILE_MIME_TYPE: {
            return "File MIME Type";
        }
        case MATROSKA_ID_FILE_DATA: {
            return "File Data";
        }
        case MATROSKA_ID_FILE_UID: {
            return "File UID";
        }
        case MATROSKA_ID_CUES: {
            return "Cues";
        }
        case MATROSKA_ID_CUE_POINT: {
            return "Cue Point";
        }
        case MATROSKA_ID_CUE_TIME: {
            return "Cue Time";
        }
        case MATROSKA_ID_CUE_TRACK_POSITIONS: {
            return "Cue Track Positions";
        }
        case MATROSKA_ID_CUE_TRACK: {
            return "Cue Track";
        }
        case MATROSKA_ID_CUE_CLUSTER_POSITION: {
            return "Cue Cluster Position";
        }
        case MATROSKA_ID_CUE_BLOCK_NUMBER: {
            return "Cue Block Number";
        }
        case MATROSKA_ID_TAGS: {
            return "Tags";
        }
        case MATROSKA_ID_SEEK_PRE_ROLL: {
            return "SeekPreRoll";
        }
        case MATROSKA_ID_CODEC_DELAY: {
            return "CodecDelay";
        }
        case MATROSKA_ID_DISCARD_PADDING: {
            return "DiscardPadding";
        }
        default: {
            return "*****unknown*****";
        }
    }
}

EBMLDataSize::EBMLDataSize() : EBMLNumber(True) {}

EBMLDataSize::~EBMLDataSize() {}
