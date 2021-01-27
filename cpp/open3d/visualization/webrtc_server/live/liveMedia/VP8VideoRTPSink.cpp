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
// RTP sink for VP8 video
// Implementation

#include "VP8VideoRTPSink.hh"

VP8VideoRTPSink ::VP8VideoRTPSink(UsageEnvironment& env,
                                  Groupsock* RTPgs,
                                  unsigned char rtpPayloadFormat)
    : VideoRTPSink(env, RTPgs, rtpPayloadFormat, 90000, "VP8") {}

VP8VideoRTPSink::~VP8VideoRTPSink() {}

VP8VideoRTPSink* VP8VideoRTPSink::createNew(UsageEnvironment& env,
                                            Groupsock* RTPgs,
                                            unsigned char rtpPayloadFormat) {
    return new VP8VideoRTPSink(env, RTPgs, rtpPayloadFormat);
}

Boolean VP8VideoRTPSink ::frameCanAppearAfterPacketStart(
        unsigned char const* /*frameStart*/,
        unsigned /*numBytesInFrame*/) const {
    // A packet can contain only one frame
    return False;
}

void VP8VideoRTPSink ::doSpecialFrameHandling(
        unsigned fragmentationOffset,
        unsigned char* /*frameStart*/,
        unsigned /*numBytesInFrame*/,
        struct timeval framePresentationTime,
        unsigned numRemainingBytes) {
    // Set the "VP8 Payload Descriptor" (just the minimal required 1-byte
    // version):
    u_int8_t vp8PayloadDescriptor = fragmentationOffset == 0 ? 0x10 : 0x00;
    // X = R = N = 0; PartID = 0; S = 1 iff this is the first (or only) fragment
    // of the frame
    setSpecialHeaderBytes(&vp8PayloadDescriptor, 1);

    if (numRemainingBytes == 0) {
        // This packet contains the last (or only) fragment of the frame.
        // Set the RTP 'M' ('marker') bit:
        setMarkerBit();
    }

    // Also set the RTP timestamp:
    setTimestamp(framePresentationTime);
}

unsigned VP8VideoRTPSink::specialHeaderSize() const {
    // We include only the required 1-byte form of the "VP8 Payload Descriptor":
    return 1;
}
