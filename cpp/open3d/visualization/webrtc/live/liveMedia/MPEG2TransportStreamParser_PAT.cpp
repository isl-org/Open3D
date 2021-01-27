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

void MPEG2TransportStreamParser::parsePAT(Boolean pusi, unsigned numDataBytes) {
#ifdef DEBUG_CONTENTS
    fprintf(stderr, "\tProgram Association Table\n");
#endif
    unsigned startPos = curOffset();

    do {
        if (pusi) {
            u_int8_t pointer_field = get1Byte();
            skipBytes(pointer_field);  // usually 0
        }

        u_int8_t table_id = get1Byte();
        if (table_id != 0x00) {
#ifdef DEBUG_ERRORS
            fprintf(stderr,
                    "MPEG2TransportStreamParser::parsePAT(%d, %d): bad "
                    "table_id: 0x%02x\n",
                    pusi, numDataBytes, table_id);
#endif
            break;
        }

        u_int16_t flagsPlusSection_length = get2Bytes();
        u_int16_t section_length = flagsPlusSection_length & 0x0FFF;
#ifdef DEBUG_CONTENTS
        fprintf(stderr, "\t\tsection_length: %d\n", section_length);
#endif
        if (section_length < 9 /*too small for remaining fields + CRC*/ ||
            section_length > 1021 /*as per specification*/) {
#ifdef DEBUG_ERRORS
            fprintf(stderr,
                    "MPEG2TransportStreamParser::parsePAT(%d, %d): Bad "
                    "section_length: %d\n",
                    pusi, numDataBytes, section_length);
#endif
            break;
        }

        unsigned endPos = curOffset() + section_length;
        if (endPos - startPos > numDataBytes) {
#ifdef DEBUG_ERRORS
            fprintf(stderr,
                    "MPEG2TransportStreamParser::parsePAT(%d, %d): "
                    "section_length %d gives us a total size %d that's too "
                    "large!\n",
                    pusi, numDataBytes, section_length, endPos - startPos);
#endif
            break;
        }

#ifdef DEBUG_CONTENTS
        u_int16_t transport_stream_id = get2Bytes();
        fprintf(stderr, "\t\ttransport_stream_id: 0x%04x\n",
                transport_stream_id);
        u_int8_t version_number_byte = get1Byte();
        u_int8_t version_number = (version_number_byte & 0x1E) >> 1;
        u_int8_t section_number = get1Byte();
        u_int8_t last_section_number = get1Byte();
        fprintf(stderr,
                "\t\tversion_number: %d; section_number: %d; "
                "last_section_number: %d\n",
                version_number, section_number, last_section_number);
#else
        skipBytes(5);
#endif

        while (curOffset() <=
               endPos - 4 /*for CRC*/ - 4 /*for a program_number+PID*/) {
            u_int16_t program_number = get2Bytes();
            u_int16_t pid = get2Bytes() & 0x1FFF;

#ifdef DEBUG_CONTENTS
            fprintf(stderr, "\t\tprogram_number: %d; PID: 0x%04x\n",
                    program_number, pid);
#endif
            if (program_number != 0x0000) {
                if (fPIDState[pid] == NULL)
                    fPIDState[pid] =
                            new PIDState_PMT(*this, pid, program_number);
            }
        }
    } while (0);

    // Skip (ignore) all remaining bytes in this packet (including the CRC):
    int numBytesLeft = numDataBytes - (curOffset() - startPos);
    if (numBytesLeft > 0) {
#ifdef DEBUG_CONTENTS
        fprintf(stderr, "\t\t+%d CRC and stuffing bytes\n", numBytesLeft);
#endif
        skipBytes(numBytesLeft);
    }
}

//########## PIDState_PAT implementation ##########

PIDState_PAT::PIDState_PAT(MPEG2TransportStreamParser& parser, u_int16_t pid)
    : PIDState(parser, pid, PAT) {}

PIDState_PAT::~PIDState_PAT() {}
