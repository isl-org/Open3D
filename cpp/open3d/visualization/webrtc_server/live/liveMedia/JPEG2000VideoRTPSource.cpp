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

#include "JPEG2000VideoRTPSource.hh"

JPEG2000VideoRTPSource* JPEG2000VideoRTPSource::createNew(
        UsageEnvironment& env,
        Groupsock* RTPgs,
        unsigned char rtpPayloadFormat,
        unsigned rtpTimestampFrequency,
        char const* sampling) {
    return new JPEG2000VideoRTPSource(env, RTPgs, rtpPayloadFormat,
                                      rtpTimestampFrequency, sampling);
}

JPEG2000VideoRTPSource ::JPEG2000VideoRTPSource(UsageEnvironment& env,
                                                Groupsock* RTPgs,
                                                unsigned char rtpPayloadFormat,
                                                unsigned rtpTimestampFrequency,
                                                char const* sampling)
    : MultiFramedRTPSource(
              env, RTPgs, rtpPayloadFormat, rtpTimestampFrequency) {
    fSampling = strDup(sampling);
}

JPEG2000VideoRTPSource::~JPEG2000VideoRTPSource() { delete[] fSampling; }

#define JPEG2000_PAYLOAD_HEADER_SIZE 8

Boolean JPEG2000VideoRTPSource ::processSpecialHeader(
        BufferedPacket* packet, unsigned& resultSpecialHeaderSize) {
    unsigned char* headerStart = packet->data();
    unsigned packetSize = packet->dataSize();

    // There should be enough space for a payload header:
    if (packetSize < JPEG2000_PAYLOAD_HEADER_SIZE) return False;

    u_int32_t fragmentOffset =
            (headerStart[5] << 16) | (headerStart[6] << 8) | (headerStart[7]);
    fCurrentPacketBeginsFrame = fragmentOffset == 0;
    fCurrentPacketCompletesFrame = packet->rtpMarkerBit();

    resultSpecialHeaderSize = JPEG2000_PAYLOAD_HEADER_SIZE;
    return True;
}

char const* JPEG2000VideoRTPSource::MIMEtype() const {
    return "video/JPEG2000";
}
