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
// Implementation

#include <time.h>  // for time_t

#include "FileSink.hh"
#include "MPEG2TransportStreamParser.hh"

Boolean MPEG2TransportStreamParser ::processStreamPacket(
        PIDState_STREAM* pidState, Boolean pusi, unsigned numDataBytes) {
#ifdef DEBUG_CONTENTS
    extern StreamType StreamTypes[];
    fprintf(stderr, "\t%s stream (stream_type 0x%02x)\n",
            StreamTypes[pidState->stream_type].description,
            pidState->stream_type);
#endif
    do {
        MPEG2TransportStreamDemuxedTrack* streamSource = pidState->streamSource;
        if (streamSource == NULL) {
            // There's no source for this track; just skip the data:
            skipBytes(numDataBytes);
            break;
        }

        if (!streamSource->isCurrentlyAwaitingData()) {
            // Wait until the source next gets read from.  (The parsing will
            // continue then.)
            return False;
        }

        // If the data begins with a PES header, parse it first
        unsigned pesHeaderSize = 0;
        if (pusi && pidState->stream_type != 0x05/*these special private streams don't have PES hdrs*/) {
            pesHeaderSize = parsePESHeader(pidState, numDataBytes);
            if (pesHeaderSize == 0) break;  // PES header parsing failed
        }

        // Deliver the data:
        unsigned numBytesToDeliver = numDataBytes - pesHeaderSize;
        if (numBytesToDeliver > streamSource->maxSize()) {
            streamSource->frameSize() = streamSource->maxSize();
            streamSource->numTruncatedBytes() =
                    numBytesToDeliver - streamSource->maxSize();
        } else {
            streamSource->frameSize() = numBytesToDeliver;
            streamSource->numTruncatedBytes() = 0;
        }
        getBytes(streamSource->to(), streamSource->frameSize());
        skipBytes(streamSource->numTruncatedBytes());

        double pts = pidState->lastSeenPTS == 0.0 ? fLastSeenPCR
                                                  : pidState->lastSeenPTS;
        streamSource->presentationTime().tv_sec = (time_t)pts;
        streamSource->presentationTime().tv_usec =
                int(pts * 1000000.0) % 1000000;

        FramedSource::afterGetting(streamSource);  // completes delivery
    } while (0);

    return True;
}

static Boolean isSpecialStreamId[0x100];

unsigned MPEG2TransportStreamParser ::parsePESHeader(PIDState_STREAM* pidState,
                                                     unsigned numDataBytes) {
    static Boolean haveInitializedIsSpecialStreamId = False;
    if (!haveInitializedIsSpecialStreamId) {
        for (unsigned i = 0; i < 0x100; ++i) isSpecialStreamId[i] = False;
        isSpecialStreamId[0xBC] = True;  // program_stream_map
        isSpecialStreamId[0xBE] = True;  // padding_stream
        isSpecialStreamId[0xBF] = True;  // private_stream_2
        isSpecialStreamId[0xF0] = True;  // ECM_stream
        isSpecialStreamId[0xF1] = True;  // EMM_stream
        isSpecialStreamId[0xF2] = True;  // DSMCC_stream
        isSpecialStreamId[0xF8] = True;  // ITU-T Rec. H.222.1 type E
        isSpecialStreamId[0xFF] = True;  // program_stream_directory

        haveInitializedIsSpecialStreamId = True;  // from now on
    }

#ifdef DEBUG_CONTENTS
    fprintf(stderr, "\t\tPES Header:\n");
#endif
    unsigned startPos = curOffset();

    do {
        u_int32_t startCodePlusStreamId = get4Bytes();
        if ((startCodePlusStreamId & 0xFFFFFF00) != 0x00000100) {
#ifdef DEBUG_ERRORS
            fprintf(stderr,
                    "MPEG2TransportStreamParser::parsePESHeader(0x%02x, %d): "
                    "Bad start code: 0x%06x\n",
                    pidState->PID, numDataBytes, startCodePlusStreamId >> 8);
#endif
            break;
        }
        u_int8_t stream_id = startCodePlusStreamId & 0xFF;

#ifdef DEBUG_CONTENTS
        fprintf(stderr, "\t\t\tstream_id: 0x%02x; PES_packet_length: %d\n",
                stream_id, get2Bytes());
#else
        skipBytes(2);
#endif

        if (!isSpecialStreamId[stream_id]) {
            u_int16_t flags = get2Bytes();
            if ((flags & 0xC000) != 0x8000) {
#ifdef DEBUG_ERRORS
                fprintf(stderr,
                        "MPEG2TransportStreamParser::parsePESHeader(0x%02x, "
                        "%d): Bad flags: 0x%04x\n",
                        pidState->PID, numDataBytes, flags);
#endif
                break;
            }
            u_int8_t PTS_DTS_flags = (flags & 0x00C0) >> 6;
            Boolean ESCR_flag = (flags & 0x0020) != 0;
            Boolean ES_rate_flag = (flags & 0x0010) != 0;
            Boolean DSM_trick_mode_flag = (flags & 0x0008) != 0;
            Boolean additional_copy_info_flag = (flags & 0x0004) != 0;
            Boolean PES_CRC_flag = (flags & 0x0002) != 0;
            Boolean PES_extension_flag = (flags & 0x0001) != 0;
#ifdef DEBUG_CONTENTS
            fprintf(stderr,
                    "\t\t\tflags: 0x%04x (PTS_DTS:%d; ESCR:%d; ES_rate:%d; "
                    "DSM_trick_mode:%d; additional_copy_info:%d; PES_CRC:%d; "
                    "PES_extension:%d)\n",
                    flags, PTS_DTS_flags, ESCR_flag, ES_rate_flag,
                    DSM_trick_mode_flag, additional_copy_info_flag,
                    PES_CRC_flag, PES_extension_flag);
#endif

            u_int8_t PES_header_data_length = get1Byte();
#ifdef DEBUG_CONTENTS
            fprintf(stderr, "\t\t\tPES_header_data_length: %d\n",
                    PES_header_data_length);
#endif

            if (PTS_DTS_flags == 2 || PTS_DTS_flags == 3) {
                // Begin with a PTS:
                u_int8_t first8PTSBits = get1Byte();
                u_int32_t last32PTSBits = get4Bytes();
                if ((first8PTSBits & 0xF1) != ((PTS_DTS_flags << 4) | 0x01) ||
                    (last32PTSBits & 0x00010001) != 0x00010001) {
#ifdef DEBUG_ERRORS
                    fprintf(stderr,
                            "MPEG2TransportStreamParser::parsePESHeader(0x%02x,"
                            " %d): Bad PTS bits: 0x%02x,0x%08x\n",
                            pidState->PID, numDataBytes, first8PTSBits,
                            last32PTSBits);
#endif
                    break;
                }
                u_int32_t ptsUpper32 = ((first8PTSBits & 0x0E) << 28) |
                                       ((last32PTSBits & 0xFFFE0000) >> 3) |
                                       ((last32PTSBits & 0x0000FFFC) >> 2);
                u_int8_t ptsLowBit = (last32PTSBits & 0x00000002) >> 1;
                double PTS = ptsUpper32 / 45000.0;
                if (ptsLowBit) PTS += 1 / 90000.0;
#ifdef DEBUG_CONTENTS
                fprintf(stderr, "\t\t\tPTS: 0x%02x%08x => 0x%08x+%d => %.10f\n",
                        first8PTSBits, last32PTSBits, ptsUpper32, ptsLowBit,
                        PTS);
#endif
                // Record this PTS:
                pidState->lastSeenPTS = PTS;
            }

            if (PTS_DTS_flags == 3) {
                // Continue with a DTS:
                u_int8_t first8DTSBits = get1Byte();
                u_int32_t last32DTSBits = get4Bytes();
                if ((first8DTSBits & 0x11) != 0x11 ||
                    (last32DTSBits & 0x00010001) != 0x00010001) {
#ifdef DEBUG_ERRORS
                    fprintf(stderr,
                            "MPEG2TransportStreamParser::parsePESHeader(0x%02x,"
                            " %d): Bad DTS bits: 0x%02x,0x%08x\n",
                            pidState->PID, numDataBytes, first8DTSBits,
                            last32DTSBits);
#endif
                    break;
                }
                u_int32_t dtsUpper32 = ((first8DTSBits & 0x0E) << 28) |
                                       ((last32DTSBits & 0xFFFE0000) >> 3) |
                                       ((last32DTSBits & 0x0000FFFC) >> 2);
                u_int8_t dtsLowBit = (last32DTSBits & 0x00000002) >> 1;
                double DTS = dtsUpper32 / 45000.0;
                if (dtsLowBit) DTS += 1 / 90000.0;
#ifdef DEBUG_CONTENTS
                fprintf(stderr, "\t\t\tDTS: 0x%02x%08x => 0x%08x+%d => %.10f\n",
                        first8DTSBits, last32DTSBits, dtsUpper32, dtsLowBit,
                        DTS);
#endif
            }

            if (ESCR_flag) {
                // Skip over the ESCR
                skipBytes(6);
            }

            if (ES_rate_flag) {
                // Skip over the ES_rate
                skipBytes(6);
            }

            if (DSM_trick_mode_flag) {
                // Skip over this
                skipBytes(1);
            }

            if (additional_copy_info_flag) {
                // Skip over this
                skipBytes(1);
            }

            if (PES_CRC_flag) {
                // Skip over this
                skipBytes(2);
            }

            if (PES_extension_flag) {
                u_int8_t flags = get1Byte();
                Boolean PES_private_data_flag = (flags & 0x80) != 0;
                Boolean pack_header_field_flag = (flags & 0x40) != 0;
                Boolean program_packet_sequence_counter_flag =
                        (flags & 0x20) != 0;
                Boolean P_STD_buffer_flag = (flags & 0x10) != 0;
                Boolean PES_extension_flag_2 = (flags & 0x01) != 0;
#ifdef DEBUG_CONTENTS
                fprintf(stderr,
                        "\t\t\tPES_extension: flags: 0x%02x "
                        "(PES_private_data:%d; pack_header_field:%d; "
                        "program_packet_sequence_counter:%d; P_STD_buffer:%d; "
                        "PES_extension_2:%d\n",
                        flags, PES_private_data_flag, pack_header_field_flag,
                        program_packet_sequence_counter_flag, P_STD_buffer_flag,
                        PES_extension_flag_2);
#endif
                if (PES_private_data_flag) {
                    // Skip over this
                    skipBytes(16);
                }
                if (pack_header_field_flag) {
                    // Skip over this
                    skipBytes(1 + 12);  // "pack_header()" is 12 bytes in size
                }
                if (program_packet_sequence_counter_flag) {
                    // Skip over this
                    skipBytes(2);
                }
                if (P_STD_buffer_flag) {
                    // Skip over this
                    skipBytes(2);
                }
                if (PES_extension_flag_2) {
                    u_int8_t PES_extension_field_length = get1Byte() & 0x7F;
#ifdef DEBUG_CONTENTS
                    fprintf(stderr, "\t\t\t\tPES_extension_field_length: %d\n",
                            PES_extension_field_length);
#endif
                    skipBytes(PES_extension_field_length);
                }
            }

            // Make sure that the number of header bytes parsed is consistent
            // with "PES_header_data_length" (and skip over any remasining
            // 'stuffing' bytes):
            if ((int)(curOffset() - startPos) > 9 + PES_header_data_length) {
#ifdef DEBUG_ERRORS
                fprintf(stderr,
                        "MPEG2TransportStreamParser::parsePESHeader(0x%02x, "
                        "%d): Error: Parsed %d PES header bytes; expected %d "
                        "(based on \"PES_header_data_length\": %d)\n",
                        pidState->PID, numDataBytes, curOffset() - startPos,
                        9 + PES_header_data_length, PES_header_data_length);
#endif
                break;
            }
            skipBytes(9 + PES_header_data_length -
                      (curOffset() - startPos));  // >= 0
        }

        unsigned PESHeaderSize = curOffset() - startPos;
#ifdef DEBUG_CONTENTS
        fprintf(stderr, "\t\t\t=> PES header size: %d\n", PESHeaderSize);
#endif
        if (PESHeaderSize > numDataBytes) {
#ifdef DEBUG_ERRORS
            fprintf(stderr,
                    "MPEG2TransportStreamParser::parsePESHeader(0x%02x, %d): "
                    "Error: PES header size %d is larger than the number of "
                    "bytes available (%d)\n",
                    pidState->PID, numDataBytes, PESHeaderSize, numDataBytes);
#endif
            break;
        }
        return PESHeaderSize;
    } while (0);

    // An error occurred.  Skip over any remaining bytes in the packet:
    int numBytesLeft = numDataBytes - (curOffset() - startPos);
    if (numBytesLeft > 0) skipBytes((unsigned)numBytesLeft);
    return 0;
}

//########## PIDState_STREAM implementation ##########

PIDState_STREAM::PIDState_STREAM(MPEG2TransportStreamParser& parser,
                                 u_int16_t pid,
                                 u_int16_t programNumber,
                                 u_int8_t streamType)
    : PIDState(parser, pid, STREAM),
      program_number(programNumber),
      stream_type(streamType),
      lastSeenPTS(0.0) {
    // Create the 'source' and 'sink' objects for this track, and 'start
    // playing' them:
    streamSource = new MPEG2TransportStreamDemuxedTrack(parser, pid);

    char fileName[100];
    extern StreamType StreamTypes[];
    StreamType& st = StreamTypes[streamType];  // alias
    sprintf(fileName, "%s-0x%04x-0x%04x%s",
            st.dataType == StreamType::AUDIO
                    ? "AUDIO"
                    : st.dataType == StreamType::VIDEO
                              ? "VIDEO"
                              : st.dataType == StreamType::DATA
                                        ? "DATA"
                                        : st.dataType == StreamType::TEXT
                                                  ? "TEXT"
                                                  : "UNKNOWN",
            program_number, pid, st.filenameSuffix);
    fprintf(stderr, "Creating new output file \"%s\"\n", fileName);
    streamSink = FileSink::createNew(parser.envir(), fileName);
    streamSink->startPlaying(*streamSource, NULL, NULL);
}

PIDState_STREAM::~PIDState_STREAM() {
    Medium::close(streamSink);
    Medium::close(streamSource);
}
