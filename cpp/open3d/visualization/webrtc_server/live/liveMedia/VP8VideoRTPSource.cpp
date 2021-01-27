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
// VP8 Video RTP Sources
// Implementation

#include "VP8VideoRTPSource.hh"

VP8VideoRTPSource* VP8VideoRTPSource::createNew(
        UsageEnvironment& env,
        Groupsock* RTPgs,
        unsigned char rtpPayloadFormat,
        unsigned rtpTimestampFrequency) {
    return new VP8VideoRTPSource(env, RTPgs, rtpPayloadFormat,
                                 rtpTimestampFrequency);
}

VP8VideoRTPSource ::VP8VideoRTPSource(UsageEnvironment& env,
                                      Groupsock* RTPgs,
                                      unsigned char rtpPayloadFormat,
                                      unsigned rtpTimestampFrequency)
    : MultiFramedRTPSource(
              env, RTPgs, rtpPayloadFormat, rtpTimestampFrequency) {}

VP8VideoRTPSource::~VP8VideoRTPSource() {}

#define incrHeader                           \
    do {                                     \
        ++resultSpecialHeaderSize;           \
        ++headerStart;                       \
        if (--packetSize == 0) return False; \
    } while (0)

Boolean VP8VideoRTPSource ::processSpecialHeader(
        BufferedPacket* packet, unsigned& resultSpecialHeaderSize) {
    unsigned char* headerStart = packet->data();
    unsigned packetSize = packet->dataSize();

    // The special header is from 1 to 6 bytes long.
    if (packetSize == 0) return False;  // error
    resultSpecialHeaderSize = 1;        // unless we learn otherwise

    u_int8_t const byte1 = *headerStart;
    Boolean const X = (byte1 & 0x80) != 0;
    Boolean const S = (byte1 & 0x10) != 0;
    u_int8_t const PartID = byte1 & 0x0F;

    fCurrentPacketBeginsFrame = S && PartID == 0;
    fCurrentPacketCompletesFrame =
            packet->rtpMarkerBit();  // RTP header's "M" bit

    if (X) {
        incrHeader;

        u_int8_t const byte2 = *headerStart;
        Boolean const I = (byte2 & 0x80) != 0;
        Boolean const L = (byte2 & 0x40) != 0;
        Boolean const T = (byte2 & 0x20) != 0;
        Boolean const K = (byte2 & 0x10) != 0;

        if (I) {
            incrHeader;
            if ((*headerStart) &
                0x80) {  // extension flag in the PictureID is set
                incrHeader;
            }
        }

        if (L) incrHeader;
        if (T || K) incrHeader;
    }

    return True;
}

char const* VP8VideoRTPSource::MIMEtype() const { return "video/VP8"; }
