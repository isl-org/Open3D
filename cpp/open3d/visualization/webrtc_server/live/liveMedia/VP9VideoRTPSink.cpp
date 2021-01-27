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
// RTP sink for VP9 video
// Implementation

#include "VP9VideoRTPSink.hh"

VP9VideoRTPSink ::VP9VideoRTPSink(UsageEnvironment& env,
                                  Groupsock* RTPgs,
                                  unsigned char rtpPayloadFormat)
    : VideoRTPSink(env, RTPgs, rtpPayloadFormat, 90000, "VP9") {}

VP9VideoRTPSink::~VP9VideoRTPSink() {}

VP9VideoRTPSink* VP9VideoRTPSink::createNew(UsageEnvironment& env,
                                            Groupsock* RTPgs,
                                            unsigned char rtpPayloadFormat) {
    return new VP9VideoRTPSink(env, RTPgs, rtpPayloadFormat);
}

Boolean VP9VideoRTPSink ::frameCanAppearAfterPacketStart(
        unsigned char const* /*frameStart*/,
        unsigned /*numBytesInFrame*/) const {
    // A packet can contain only one frame
    return False;
}

void VP9VideoRTPSink ::doSpecialFrameHandling(
        unsigned fragmentationOffset,
        unsigned char* /*frameStart*/,
        unsigned /*numBytesInFrame*/,
        struct timeval framePresentationTime,
        unsigned numRemainingBytes) {
    // Set the "VP9 Payload Descriptor" (just the minimal required 1-byte
    // version):
    u_int8_t vp9PayloadDescriptor = fragmentationOffset == 0 ? 0x10 : 0x00;
    // I = L = F = V = U = 0; S = 1 iff this is the first (or only) fragment of
    // the frame

    if (numRemainingBytes == 0) {
        // This packet contains the last (or only) fragment of the frame.
        // Set the E bit:
        vp9PayloadDescriptor |= 0x08;
        // Also set the RTP 'M' ('marker') bit:
        setMarkerBit();
    }

    setSpecialHeaderBytes(&vp9PayloadDescriptor, 1);

    // Also set the RTP timestamp:
    setTimestamp(framePresentationTime);
}

unsigned VP9VideoRTPSink::specialHeaderSize() const {
    // We include only the required 1-byte form of the "VP9 Payload Descriptor":
    return 1;
}
