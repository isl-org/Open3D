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
// Theora Video RTP Sources
// Implementation

#include "TheoraVideoRTPSource.hh"

////////// TheoraBufferedPacket and TheoraBufferedPacketFactory //////////

class TheoraBufferedPacket : public BufferedPacket {
public:
    TheoraBufferedPacket();
    virtual ~TheoraBufferedPacket();

private:  // redefined virtual functions
    virtual unsigned nextEnclosedFrameSize(unsigned char*& framePtr,
                                           unsigned dataSize);
};

class TheoraBufferedPacketFactory : public BufferedPacketFactory {
private:  // redefined virtual functions
    virtual BufferedPacket* createNewPacket(MultiFramedRTPSource* ourSource);
};

///////// TheoraVideoRTPSource implementation ////////

TheoraVideoRTPSource* TheoraVideoRTPSource::createNew(
        UsageEnvironment& env,
        Groupsock* RTPgs,
        unsigned char rtpPayloadFormat) {
    return new TheoraVideoRTPSource(env, RTPgs, rtpPayloadFormat);
}

TheoraVideoRTPSource ::TheoraVideoRTPSource(UsageEnvironment& env,
                                            Groupsock* RTPgs,
                                            unsigned char rtpPayloadFormat)
    : MultiFramedRTPSource(env,
                           RTPgs,
                           rtpPayloadFormat,
                           90000,
                           new TheoraBufferedPacketFactory),
      fCurPacketIdent(0) {}

TheoraVideoRTPSource::~TheoraVideoRTPSource() {}

Boolean TheoraVideoRTPSource ::processSpecialHeader(
        BufferedPacket* packet, unsigned& resultSpecialHeaderSize) {
    unsigned char* headerStart = packet->data();
    unsigned packetSize = packet->dataSize();

    resultSpecialHeaderSize = 4;
    if (packetSize < resultSpecialHeaderSize)
        return False;  // packet was too small

    // The first 3 bytes of the header are the "Ident" field:
    fCurPacketIdent =
            (headerStart[0] << 16) | (headerStart[1] << 8) | headerStart[2];

    // The 4th byte is F|TDT|numPkts.
    // Reject any packet with TDT == 3:
    if ((headerStart[3] & 0x30) == 0x30) return False;

    u_int8_t F = headerStart[3] >> 6;
    fCurrentPacketBeginsFrame = F <= 1;  // "Not Fragmented" or "Start Fragment"
    fCurrentPacketCompletesFrame =
            F == 0 || F == 3;  // "Not Fragmented" or "End Fragment"

    return True;
}

char const* TheoraVideoRTPSource::MIMEtype() const { return "video/THEORA"; }

////////// TheoraBufferedPacket and TheoraBufferedPacketFactory implementation
/////////////

TheoraBufferedPacket::TheoraBufferedPacket() {}

TheoraBufferedPacket::~TheoraBufferedPacket() {}

unsigned TheoraBufferedPacket ::nextEnclosedFrameSize(unsigned char*& framePtr,
                                                      unsigned dataSize) {
    if (dataSize < 2) {
        // There's not enough space for a 2-byte header.  TARFU!  Just return
        // the data that's left:
        return dataSize;
    }

    unsigned frameSize = (framePtr[0] << 8) | framePtr[1];
    framePtr += 2;
    if (frameSize > dataSize - 2)
        return dataSize - 2;  // inconsistent frame size => just return all the
                              // data that's left

    return frameSize;
}

BufferedPacket* TheoraBufferedPacketFactory ::createNewPacket(
        MultiFramedRTPSource* /*ourSource*/) {
    return new TheoraBufferedPacket();
}
