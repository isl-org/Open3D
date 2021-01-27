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

#include "JPEG2000VideoRTPSink.hh"

JPEG2000VideoRTPSink::JPEG2000VideoRTPSink(UsageEnvironment& env,
                                           Groupsock* RTPgs)
    : VideoRTPSink(env, RTPgs, 98, 90000, "jpeg2000") {}

JPEG2000VideoRTPSink::~JPEG2000VideoRTPSink() {}

JPEG2000VideoRTPSink* JPEG2000VideoRTPSink::createNew(UsageEnvironment& env,
                                                      Groupsock* RTPgs) {
    return new JPEG2000VideoRTPSink(env, RTPgs);
}

#define JPEG2000_PAYLOAD_HEADER_SIZE 8

void JPEG2000VideoRTPSink ::doSpecialFrameHandling(
        unsigned fragmentationOffset,
        unsigned char* frameStart,
        unsigned numBytesInFrame,
        struct timeval framePresentationTime,
        unsigned numRemainingBytes) {
    // Fill in the Payload Header:
    u_int8_t payloadHeader[JPEG2000_PAYLOAD_HEADER_SIZE];

    // For "tp", assume for now that the payload is progressively scanned (i.e.,
    // tp = 0) For "MHF", assume that a whole main header is present (i.e., MHF
    // = 3), *unless* we're
    //   the second or later packet of a fragment, in which case we assume that
    //   it's not (i.e. MHF = 0)
    // For "mh_id", set this to 0 (as specified in RFC 5371).
    // For "T" (Tile field invalidation flag), set this to 0 (we don't set the
    // "tile number" field).
    payloadHeader[0] = fragmentationOffset > 0 ? 0x00 : 0x30;

    // Set the "priority" field to 255, as specified in RFC 5371:
    payloadHeader[1] = 255;

    // Set the "tile number" field to 0:
    payloadHeader[2] = payloadHeader[3] = 0;

    // Set the "reserved" field to 0, as specified in RFC 5371:
    payloadHeader[4] = 0;

    // Set the "fragmentation offset" field to the value of our
    // "fragmentationOffset" parameter:
    payloadHeader[5] = (u_int8_t)(fragmentationOffset >> 16);
    payloadHeader[6] = (u_int8_t)(fragmentationOffset >> 8);
    payloadHeader[7] = (u_int8_t)(fragmentationOffset);

    // Write the payload header to the outgoing packet:
    setSpecialHeaderBytes(payloadHeader, sizeof payloadHeader);

    if (numRemainingBytes == 0) {
        // This packet contains the last (or only) fragment of the frame.
        // Set the RTP 'M' ('marker') bit
        setMarkerBit();
    }

    // Also set the RTP timestamp:
    setTimestamp(framePresentationTime);
}

unsigned JPEG2000VideoRTPSink::specialHeaderSize() const {
    return JPEG2000_PAYLOAD_HEADER_SIZE;
}
