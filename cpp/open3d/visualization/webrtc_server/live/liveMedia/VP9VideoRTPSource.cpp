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
// VP9 Video RTP Sources
// Implementation

#include "VP9VideoRTPSource.hh"

VP9VideoRTPSource* VP9VideoRTPSource::createNew(
        UsageEnvironment& env,
        Groupsock* RTPgs,
        unsigned char rtpPayloadFormat,
        unsigned rtpTimestampFrequency) {
    return new VP9VideoRTPSource(env, RTPgs, rtpPayloadFormat,
                                 rtpTimestampFrequency);
}

VP9VideoRTPSource ::VP9VideoRTPSource(UsageEnvironment& env,
                                      Groupsock* RTPgs,
                                      unsigned char rtpPayloadFormat,
                                      unsigned rtpTimestampFrequency)
    : MultiFramedRTPSource(
              env, RTPgs, rtpPayloadFormat, rtpTimestampFrequency) {}

VP9VideoRTPSource::~VP9VideoRTPSource() {}

#define incrHeader                           \
    do {                                     \
        ++resultSpecialHeaderSize;           \
        ++headerStart;                       \
        if (--packetSize == 0) return False; \
    } while (0)

Boolean VP9VideoRTPSource ::processSpecialHeader(
        BufferedPacket* packet, unsigned& resultSpecialHeaderSize) {
    unsigned char* headerStart = packet->data();
    unsigned packetSize = packet->dataSize();

    // Figure out the size of the special header.
    if (packetSize == 0) return False;  // error
    resultSpecialHeaderSize = 1;        // unless we learn otherwise

    u_int8_t const byte1 = *headerStart;
    Boolean const I = (byte1 & 0x80) != 0;
    Boolean const L = (byte1 & 0x40) != 0;
    Boolean const F = (byte1 & 0x20) != 0;
    Boolean const B = (byte1 & 0x10) != 0;
    Boolean const E = (byte1 & 0x08) != 0;
    Boolean const V = (byte1 & 0x04) != 0;
    Boolean const U = (byte1 & 0x02) != 0;

    fCurrentPacketBeginsFrame = B;
    fCurrentPacketCompletesFrame = E;
    // use this instead of the RTP header's 'M' bit (which might not be
    // accurate)

    if (I) {  // PictureID present
        incrHeader;
        Boolean const M = ((*headerStart) & 0x80) != 0;
        if (M) incrHeader;
    }

    if (L) {  // Layer indices present
        incrHeader;
        if (F) {  // Reference indices present
            incrHeader;
            unsigned R = (*headerStart) & 0x03;
            while (R-- > 0) {
                incrHeader;
                Boolean const X = ((*headerStart) & 0x10) != 0;
                if (X) incrHeader;
            }
        }
    }

    if (V) {  // Scalability Structure (SS) present
        incrHeader;
        unsigned patternLength = *headerStart;
        while (patternLength-- > 0) {
            incrHeader;
            unsigned R = (*headerStart) & 0x03;
            while (R-- > 0) {
                incrHeader;
                Boolean const X = ((*headerStart) & 0x10) != 0;
                if (X) incrHeader;
            }
        }
    }

    if (U) {           // Scalability Structure Update (SU) present
        return False;  // This structure isn't yet defined in the VP9 payload
                       // format I-D
    }

    return True;
}

char const* VP9VideoRTPSource::MIMEtype() const { return "video/VP9"; }
