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
// AC3 Audio RTP Sources
// Implementation

#include "AC3AudioRTPSource.hh"

AC3AudioRTPSource* AC3AudioRTPSource::createNew(
        UsageEnvironment& env,
        Groupsock* RTPgs,
        unsigned char rtpPayloadFormat,
        unsigned rtpTimestampFrequency) {
    return new AC3AudioRTPSource(env, RTPgs, rtpPayloadFormat,
                                 rtpTimestampFrequency);
}

AC3AudioRTPSource::AC3AudioRTPSource(UsageEnvironment& env,
                                     Groupsock* rtpGS,
                                     unsigned char rtpPayloadFormat,
                                     unsigned rtpTimestampFrequency)
    : MultiFramedRTPSource(
              env, rtpGS, rtpPayloadFormat, rtpTimestampFrequency) {}

AC3AudioRTPSource::~AC3AudioRTPSource() {}

Boolean AC3AudioRTPSource ::processSpecialHeader(
        BufferedPacket* packet, unsigned& resultSpecialHeaderSize) {
    unsigned char* headerStart = packet->data();
    unsigned packetSize = packet->dataSize();

    // There's a 2-byte payload header at the beginning:
    if (packetSize < 2) return False;
    resultSpecialHeaderSize = 2;

    unsigned char FT = headerStart[0] & 0x03;
    fCurrentPacketBeginsFrame = FT != 3;

    // The RTP "M" (marker) bit indicates the last fragment of a frame.
    // In case the sender did not set the "M" bit correctly, we also test for FT
    // == 0:
    fCurrentPacketCompletesFrame = packet->rtpMarkerBit() || FT == 0;

    return True;
}

char const* AC3AudioRTPSource::MIMEtype() const { return "audio/AC3"; }
