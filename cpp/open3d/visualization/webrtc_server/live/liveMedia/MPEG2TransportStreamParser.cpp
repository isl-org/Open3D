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

#include "MPEG2TransportStreamParser.hh"

#define NUM_PIDS 0x10000

StreamType StreamTypes[0x100];

MPEG2TransportStreamParser ::MPEG2TransportStreamParser(
        FramedSource* inputSource,
        FramedSource::onCloseFunc* onEndFunc,
        void* onEndClientData)
    : StreamParser(
              inputSource, onEndFunc, onEndClientData, continueParsing, this),
      fInputSource(inputSource),
      fAmCurrentlyParsing(False),
      fOnEndFunc(onEndFunc),
      fOnEndClientData(onEndClientData),
      fLastSeenPCR(0.0) {
    if (StreamTypes[0x01].dataType ==
        StreamType::UNKNOWN) {  // initialize array with known values
        StreamTypes[0x01] =
                StreamType("MPEG-1 video", StreamType::VIDEO, ".mpv");
        StreamTypes[0x02] =
                StreamType("MPEG-2 video", StreamType::VIDEO, ".mpv");
        StreamTypes[0x03] =
                StreamType("MPEG-1 audio", StreamType::AUDIO, ".mpa");
        StreamTypes[0x04] =
                StreamType("MPEG-2 audio", StreamType::AUDIO, ".mpa");
        StreamTypes[0x05] =
                StreamType("privately-defined data", StreamType::DATA);
        StreamTypes[0x06] =
                StreamType("privately-defined data", StreamType::DATA);
        StreamTypes[0x0F] = StreamType("AAC audio", StreamType::AUDIO, ".aac");
        StreamTypes[0x10] = StreamType("MPEG-4 H.263 based video",
                                       StreamType::VIDEO, ".mpv");
        StreamTypes[0x1B] =
                StreamType("H.264 video", StreamType::VIDEO, ".h264");
        StreamTypes[0x1C] =
                StreamType("MPEG-4 raw audio", StreamType::AUDIO, ".mpa");
        StreamTypes[0x1D] = StreamType("MPEG-4 text", StreamType::TEXT, ".txt");
        StreamTypes[0x21] =
                StreamType("JPEG 2000 video", StreamType::VIDEO, ".mjpg");
        StreamTypes[0x24] =
                StreamType("H.265 video", StreamType::VIDEO, ".h265");
        StreamTypes[0x81] = StreamType("AC-3 audio", StreamType::AUDIO, ".ac3");
    }

    // Create our 'PID state' array:
    fPIDState = new PIDState*[NUM_PIDS];
    for (unsigned i = 0; i < NUM_PIDS; ++i) fPIDState[i] = NULL;

    // Initially, the only PID we know is 0x0000: a Program Association Table:
    fPIDState[0x0000] = new PIDState_PAT(*this, 0x0000);

    // Begin parsing:
    continueParsing();
}

MPEG2TransportStreamParser::~MPEG2TransportStreamParser() {
    for (unsigned i = 0; i < NUM_PIDS; ++i) delete fPIDState[i];
    delete[] fPIDState;
}

UsageEnvironment& MPEG2TransportStreamParser::envir() {
    return fInputSource->envir();
}

void MPEG2TransportStreamParser ::continueParsing(
        void* clientData,
        unsigned char* ptr,
        unsigned size,
        struct timeval presentationTime) {
    ((MPEG2TransportStreamParser*)clientData)->continueParsing();
}

void MPEG2TransportStreamParser::continueParsing() {
    if (fAmCurrentlyParsing) return;  // don't allow recursive calls to parse()

    if (fInputSource != NULL) {
        fAmCurrentlyParsing = True;
        Boolean parseSucceeded = parse();
        fAmCurrentlyParsing = False;

        if (!parseSucceeded) {
            // We didn't complete the parsing, because we had to read more data
            // from the source, or because we're waiting for another read from
            // downstream. Once that happens, we'll get called again.
            return;
        }
    }

    // We successfully parsed the file.  Call our 'done' function now:
    if (fOnEndFunc != NULL) (*fOnEndFunc)(fOnEndClientData);
}

#define TRANSPORT_SYNC_BYTE 0x47
#define TRANSPORT_PACKET_SIZE 188

Boolean MPEG2TransportStreamParser::parse() {
    if (fInputSource->isCurrentlyAwaitingData()) return False;
    // Our input source is currently being read. Wait until that read completes

    try {
        while (1) {
            // Make sure we start with a 'sync byte':
            do {
                saveParserState();
            } while (get1Byte() != TRANSPORT_SYNC_BYTE);

            // Parse and process each (remaining 187 bytes of a) 'Transport
            // Stream Packet' at a time. (Because these are a lot smaller than
            // the "StreamParser" BANK_SIZE, we don't save
            //  parser state in the middle of processing each such 'Transport
            //  Stream Packet'. Therefore, processing of each 'Transport Stream
            //  Packet' needs to be idempotent.)

            u_int16_t flagsPlusPID = get2Bytes();
            // Check the "transport_error_indicator" flag; reject the packet if
            // it's set:
            if ((flagsPlusPID & 0x8000) != 0) {
#ifdef DEBUG_ERRORS
                fprintf(stderr,
                        "MPEG2TransportStreamParser::parse() Rejected packet "
                        "with \"transport_error_indicator\" flag set!\n");
#endif
                continue;
            }
            Boolean pusi = (flagsPlusPID & 0x4000) !=
                           0;  // payload_unit_start_indicator
            // Ignore "transport_priority"
            u_int16_t PID = flagsPlusPID & 0x1FFF;
#ifdef DEBUG_CONTENTS
            fprintf(stderr,
                    "\nTransport Packet: payload_unit_start_indicator: %d; "
                    "PID: 0x%04x\n",
                    pusi, PID);
#endif

            u_int8_t controlPlusContinuity_counter = get1Byte();
            // Reject any packets where the "transport_scrambling_control" field
            // is not zero:
            if ((controlPlusContinuity_counter & 0xC0) != 0) {
#ifdef DEBUG_ERRORS
                fprintf(stderr,
                        "MPEG2TransportStreamParser::parse() Rejected packet "
                        "with \"transport_scrambling_control\" set to non-zero "
                        "value %d!\n",
                        (controlPlusContinuity_counter & 0xC0) >> 6);
#endif
                continue;
            }
            u_int8_t adaptation_field_control =
                    (controlPlusContinuity_counter & 0x30) >> 4;  // 2 bits
#ifdef DEBUG_CONTENTS
            u_int8_t continuity_counter =
                    (controlPlusContinuity_counter & 0x0F);  // 4 bits
            fprintf(stderr,
                    "adaptation_field_control: %d; continuity_counter: 0x%X\n",
                    adaptation_field_control, continuity_counter);
#endif

            u_int8_t totalAdaptationFieldSize =
                    adaptation_field_control < 2 ? 0 : parseAdaptationField();
#ifdef DEBUG_ERRORS
            if (adaptation_field_control == 2 &&
                totalAdaptationFieldSize != 1 + 183) {
                fprintf(stderr,
                        "MPEG2TransportStreamParser::parse() Warning: Got an "
                        "inconsistent \"totalAdaptationFieldSize\" %d for "
                        "adaptation_field_control == 2\n",
                        totalAdaptationFieldSize);
            }
#endif

            int numDataBytes =
                    TRANSPORT_PACKET_SIZE - 4 - totalAdaptationFieldSize;
            if (numDataBytes > 0) {
#ifdef DEBUG_CONTENTS
                fprintf(stderr, "+%d data bytes:\n", numDataBytes);
#endif
                if (!processDataBytes(PID, pusi, numDataBytes)) {
                    // The parsing got deferred (to be resumed later when a
                    // pending read happens)
                    restoreSavedParserState();  // so that we later resume
                                                // parsing at the start of the
                                                // packet
                    return False;
                }
            }
        }
    } catch (int /*e*/) {
#ifdef DEBUG_CONTENTS
        fprintf(stderr,
                "MPEG2TransportStreamParser::parse() EXCEPTION (This is normal "
                "behavior - *not* an error)\n");
#endif
        return False;  // the parsing got interrupted
    }
}

u_int8_t MPEG2TransportStreamParser::parseAdaptationField() {
    unsigned startPos = curOffset();
#ifdef DEBUG_CONTENTS
    fprintf(stderr, "\tAdaptation Field:\n");
#endif
    u_int8_t adaptation_field_length = get1Byte();
#ifdef DEBUG_CONTENTS
    fprintf(stderr, "\t\tadaptation_field_length: %d\n",
            adaptation_field_length);
#endif
    if (adaptation_field_length > 0) {
        u_int8_t flags = get1Byte();
#ifdef DEBUG_CONTENTS
        fprintf(stderr, "\t\tadaptation field flags: 0x%02x\n", flags);
#endif
        if ((flags & 0x10) != 0) {  // PCR_flag
            u_int32_t first32PCRBits = get4Bytes();
            u_int16_t last16PCRBits = get2Bytes();
            // program_clock_reference_base = "first32PCRBits" and high bit of
            // "last16PCRBits" (33 bits) program_clock_reference_extension =
            // last 9 bits of "last16PCRBits" (9 bits)
            double PCR = first32PCRBits / 45000.0;
            if ((last16PCRBits & 0x8000) != 0)
                PCR += 1 / 90000.0;  // add in low-bit (if set)
            PCR += (last16PCRBits & 0x01FF) / 27000000.0;  // add in extension
#ifdef DEBUG_CONTENTS
            fprintf(stderr, "\t\tPCR: %.10f\n", PCR);
#endif
        }
        if ((flags & 0x08) != 0) {  // OPCR_flag
            u_int32_t first32OPCRBits = get4Bytes();
            u_int16_t last16OPCRBits = get2Bytes();
            // original_program_clock_reference_base = "first32OPCRBits" and
            // high bit of "last16OPCRBits" (33 bits)
            // original_program_clock_reference_extension = last 9 bits of
            // "last16OPCRBits" (9 bits)
            double OPCR = first32OPCRBits / 45000.0;
            if ((last16OPCRBits & 0x8000) != 0)
                OPCR += 1 / 90000.0;  // add in low-bit (if set)
            OPCR += (last16OPCRBits & 0x01FF) / 27000000.0;  // add in extension
#ifdef DEBUG_CONTENTS
            fprintf(stderr, "\t\tOPCR: %.10f\n", OPCR);
#endif
        }
        if ((flags & 0x04) != 0) {  // splicing_point_flag
            skipBytes(1);           // splice_countdown
        }
        if ((flags & 0x02) != 0) {  // transport_private_data_flag
            u_int8_t transport_private_data_length = get1Byte();
#ifdef DEBUG_CONTENTS
            fprintf(stderr, "\t\ttransport_private_data_length: %d\n",
                    transport_private_data_length);
#endif
            skipBytes(transport_private_data_length);  // "private_data_byte"s
        }
        if ((flags & 0x01) != 0) {  // adaptation_field_extension_flag
#ifdef DEBUG_CONTENTS
            u_int8_t adaptation_field_extension_length = get1Byte();
            fprintf(stderr, "\t\tadaptation_field_extension_length: %d\n",
                    adaptation_field_extension_length);
#else
            skipBytes(1);  // adaptation_field_extension_length
#endif
            u_int8_t flagsPlusReserved = get1Byte();
#ifdef DEBUG_CONTENTS
            fprintf(stderr, "\t\t\tflagsPlusReserved: 0x%02x\n",
                    flagsPlusReserved);
#endif
            if ((flagsPlusReserved & 0x80) != 0) {  // ltw_flag
                skipBytes(2);  // "ltw_valid_flag" + "ltw_offset"
            }
            if ((flagsPlusReserved & 0x40) != 0) {  // piecewise_rate_flag
                skipBytes(3);  // reserved + "piecewise_rate"
            }
            if ((flagsPlusReserved & 0x20) != 0) {  // seamless_splice_flag
                skipBytes(5);                       // DTS_next_...
            }
            // Skip reserved bytes to the end of the adaptation_field:
            int numBytesLeft =
                    (1 + adaptation_field_length) - (curOffset() - startPos);
            if (numBytesLeft > 0) {
#ifdef DEBUG_CONTENTS
                fprintf(stderr, "\t\t+%d reserved bytes\n", numBytesLeft);
#endif
                skipBytes(numBytesLeft);
            }
        }
        // Skip "stuffing_byte"s to the end of the adaptation_field:
        int numBytesLeft =
                (1 + adaptation_field_length) - (curOffset() - startPos);
        if (numBytesLeft > 0) {
#ifdef DEBUG_CONTENTS
            fprintf(stderr, "\t\t+%d stuffing bytes\n", numBytesLeft);
#endif
#ifdef DEBUG_ERRORS
            for (int i = 0; i < numBytesLeft; ++i) {
                if (get1Byte() != 0xFF) {
                    fprintf(stderr,
                            "WARNING: non-stuffing byte in adaptation_field\n");
                }
            }
#else
            skipBytes(numBytesLeft);
#endif
        }
    }

    // Finally, figure out how many bytes we parsed, and compare it to what we
    // expected:
    unsigned totalAdaptationFieldSize = curOffset() - startPos;
#ifdef DEBUG_ERRORS
    if (totalAdaptationFieldSize != 1 + adaptation_field_length) {
        fprintf(stderr,
                "MPEG2TransportStreamParser::parseAdaptationField() Warning: "
                "Got an inconsistent \"totalAdaptationFieldSize\" %d; expected "
                "%d\n",
                totalAdaptationFieldSize, 1 + adaptation_field_length);
    }
#endif
    return totalAdaptationFieldSize;
}

Boolean MPEG2TransportStreamParser ::processDataBytes(u_int16_t PID,
                                                      Boolean pusi,
                                                      unsigned numDataBytes) {
    PIDState* pidState = fPIDState[PID];

    if (pidState == NULL) {  // unknown PID
#ifdef DEBUG_CONTENTS
        fprintf(stderr, "\tUnknown PID\n");
#endif
        skipBytes(numDataBytes);
        return True;
    }

    switch (pidState->type) {
        case PAT: {
            parsePAT(pusi, numDataBytes);
            return True;
        }
        case PMT: {
            parsePMT((PIDState_PMT*)pidState, pusi, numDataBytes);
            return True;
        }
        case STREAM: {
            return processStreamPacket((PIDState_STREAM*)pidState, pusi,
                                       numDataBytes);
        }
        default: {  // Never reached, but eliminates a possible error with dumb
                    // compilers
            return False;
        }
    }
}

void MPEG2TransportStreamParser::restoreSavedParserState() {
    StreamParser::restoreSavedParserState();
    fAmCurrentlyParsing = False;
}

//########## PIDState implementation ##########

PIDState::PIDState(MPEG2TransportStreamParser& parser,
                   u_int16_t pid,
                   PIDType pidType)
    : ourParser(parser), PID(pid), type(pidType) {}

PIDState::~PIDState() {}

//######### StreamType implementation ########

StreamType ::StreamType(char const* description,
                        enum dataType dataType,
                        char const* filenameSuffix)
    : description(description),
      dataType(dataType),
      filenameSuffix(filenameSuffix) {}
