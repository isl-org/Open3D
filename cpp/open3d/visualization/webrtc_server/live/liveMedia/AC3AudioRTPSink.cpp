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
// RTP sink for AC3 audio
// Implementation

#include "AC3AudioRTPSink.hh"

AC3AudioRTPSink::AC3AudioRTPSink(UsageEnvironment& env,
                                 Groupsock* RTPgs,
                                 u_int8_t rtpPayloadFormat,
                                 u_int32_t rtpTimestampFrequency)
    : AudioRTPSink(env, RTPgs, rtpPayloadFormat, rtpTimestampFrequency, "AC3"),
      fTotNumFragmentsUsed(0) {}

AC3AudioRTPSink::~AC3AudioRTPSink() {}

AC3AudioRTPSink* AC3AudioRTPSink::createNew(UsageEnvironment& env,
                                            Groupsock* RTPgs,
                                            u_int8_t rtpPayloadFormat,
                                            u_int32_t rtpTimestampFrequency) {
    return new AC3AudioRTPSink(env, RTPgs, rtpPayloadFormat,
                               rtpTimestampFrequency);
}

Boolean AC3AudioRTPSink ::frameCanAppearAfterPacketStart(
        unsigned char const* /*frameStart*/,
        unsigned /*numBytesInFrame*/) const {
    // (For now) allow at most 1 frame in a single packet:
    return False;
}

void AC3AudioRTPSink ::doSpecialFrameHandling(
        unsigned fragmentationOffset,
        unsigned char* frameStart,
        unsigned numBytesInFrame,
        struct timeval framePresentationTime,
        unsigned numRemainingBytes) {
    // Set the 2-byte "payload header", as defined in RFC 4184.
    unsigned char headers[2];

    Boolean isFragment = numRemainingBytes > 0 || fragmentationOffset > 0;
    if (!isFragment) {
        headers[0] = 0;  // One or more complete frames
        headers[1] =
                1;  // because we (for now) allow at most 1 frame per packet
    } else {
        if (fragmentationOffset > 0) {
            headers[0] = 3;  // Fragment of frame other than initial fragment
        } else {
            // An initial fragment of the frame
            unsigned const totalFrameSize =
                    fragmentationOffset + numBytesInFrame + numRemainingBytes;
            unsigned const fiveEighthsPoint =
                    totalFrameSize / 2 + totalFrameSize / 8;
            headers[0] = numBytesInFrame >= fiveEighthsPoint ? 1 : 2;

            // Because this outgoing packet will be full (because it's an
            // initial fragment), we can compute how many total fragments (and
            // thus packets) will make up the complete AC-3 frame:
            fTotNumFragmentsUsed =
                    (totalFrameSize + (numBytesInFrame - 1)) / numBytesInFrame;
        }

        headers[1] = fTotNumFragmentsUsed;
    }

    setSpecialHeaderBytes(headers, sizeof headers);

    if (numRemainingBytes == 0) {
        // This packet contains the last (or only) fragment of the frame.
        // Set the RTP 'M' ('marker') bit:
        setMarkerBit();
    }

    // Important: Also call our base class's doSpecialFrameHandling(),
    // to set the packet's timestamp:
    MultiFramedRTPSink::doSpecialFrameHandling(
            fragmentationOffset, frameStart, numBytesInFrame,
            framePresentationTime, numRemainingBytes);
}

unsigned AC3AudioRTPSink::specialHeaderSize() const { return 2; }
