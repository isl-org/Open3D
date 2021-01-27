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
// Raw Video RTP Sources (RFC 4175)
// Implementation

#include "RawVideoRTPSource.hh"

////////// RawVideoBufferedPacket and RawVideoBufferedPacketFactory //////////

class RawVideoBufferedPacket : public BufferedPacket {
public:
    RawVideoBufferedPacket(RawVideoRTPSource* ourSource);
    virtual ~RawVideoBufferedPacket();

private:  // redefined virtual functions
    virtual void getNextEnclosedFrameParameters(
            unsigned char*& framePtr,
            unsigned dataSize,
            unsigned& frameSize,
            unsigned& frameDurationInMicroseconds);

private:
    RawVideoRTPSource* fOurSource;
};

class RawVideoBufferedPacketFactory : public BufferedPacketFactory {
private:  // redefined virtual functions
    virtual BufferedPacket* createNewPacket(MultiFramedRTPSource* ourSource);
};

////////// LineHeader //////////

struct LineHeader {
    u_int16_t length;
    u_int16_t fieldIdAndLineNumber;
    u_int16_t offsetWithinLine;
};

///////// RawVideoRTPSource implementation (RFC 4175) ////////

RawVideoRTPSource* RawVideoRTPSource::createNew(
        UsageEnvironment& env,
        Groupsock* RTPgs,
        unsigned char rtpPayloadFormat,
        unsigned rtpTimestampFrequency) {
    return new RawVideoRTPSource(env, RTPgs, rtpPayloadFormat,
                                 rtpTimestampFrequency);
}

RawVideoRTPSource ::RawVideoRTPSource(UsageEnvironment& env,
                                      Groupsock* RTPgs,
                                      unsigned char rtpPayloadFormat,
                                      unsigned rtpTimestampFrequency)
    : MultiFramedRTPSource(env,
                           RTPgs,
                           rtpPayloadFormat,
                           rtpTimestampFrequency,
                           new RawVideoBufferedPacketFactory),
      fNumLines(0),
      fNextLine(0),
      fLineHeaders(NULL) {}

RawVideoRTPSource::~RawVideoRTPSource() { delete[] fLineHeaders; }

u_int16_t RawVideoRTPSource::currentLineNumber() const {
    if (fNextLine == 0 || fLineHeaders == NULL)
        return 0;  // we've called this function too soon!
    return fLineHeaders[fNextLine - 1].fieldIdAndLineNumber & 0x7FFF;
}

u_int8_t RawVideoRTPSource::currentLineFieldId() const {
    if (fNextLine == 0 || fLineHeaders == NULL)
        return 0;  // we've called this function too soon!
    return (fLineHeaders[fNextLine - 1].fieldIdAndLineNumber & 0x8000) >> 15;
}

u_int16_t RawVideoRTPSource::currentOffsetWithinLine() const {
    if (fNextLine == 0 || fLineHeaders == NULL)
        return 0;  // we've called this function too soon!
    return fLineHeaders[fNextLine - 1].offsetWithinLine;
}

Boolean RawVideoRTPSource ::processSpecialHeader(
        BufferedPacket* packet, unsigned& resultSpecialHeaderSize) {
    unsigned char* headerStart = packet->data();
    unsigned packetSize = packet->dataSize();

    // The first 2 bytes of the header are the "Extended Sequence Number".
    // In the current implementation, we ignore this.
    if (packetSize < 2) return False;
    headerStart += 2;
    unsigned char* lineHeaderStart = headerStart;
    packetSize -= 2;

    // The rest of the header should consist of N*6 bytes (with N >= 1) for each
    // line included. Count how many of these there are:
    unsigned numLines = 0;
    while (1) {
        if (packetSize < 6)
            return False;  // there's not enough room for another line header
        ++numLines;
        Boolean continuationBit = (headerStart[4] & 0x80) >> 7;
        headerStart += 6;
        packetSize -= 6;

        // Check the "C" (continuation) bit of this header to see whether any
        // more line headers follow:
        if (continuationBit == 0) break;  // no more line headers follow
    }

    // We now know how many lines are contained in this payload.  Allocate and
    // fill in "fLineHeaders":
    fNumLines = numLines;  // ASSERT: >= 1
    fNextLine = 0;
    delete[] fLineHeaders;
    fLineHeaders = new LineHeader[fNumLines];
    unsigned totalLength = 0;
    for (unsigned i = 0; i < fNumLines; ++i) {
        fLineHeaders[i].length = (lineHeaderStart[0] << 8) + lineHeaderStart[1];
        totalLength += fLineHeaders[i].length;
        fLineHeaders[i].fieldIdAndLineNumber =
                (lineHeaderStart[2] << 8) + lineHeaderStart[3];
        fLineHeaders[i].offsetWithinLine =
                ((lineHeaderStart[4] & 0x7F) << 8) + lineHeaderStart[5];
        lineHeaderStart += 6;
    }

    // Make sure that we have enough bytes for all of the line lengths promised:
    if (totalLength > packetSize) {
        fNumLines = 0;
        delete[] fLineHeaders;
        fLineHeaders = NULL;
        return False;
    }

    // Everything looks good:
    fCurrentPacketBeginsFrame =
            (fLineHeaders[0].fieldIdAndLineNumber & 0x7FFF) == 0 &&
            fLineHeaders[0].offsetWithinLine == 0;
    fCurrentPacketCompletesFrame = packet->rtpMarkerBit();
    resultSpecialHeaderSize = headerStart - packet->data();
    return True;
}

char const* RawVideoRTPSource::MIMEtype() const { return "video/RAW"; }

////////// RawVideoBufferedPacket and RawVideoBufferedPacketFactory
/// implementation //////////

RawVideoBufferedPacket ::RawVideoBufferedPacket(RawVideoRTPSource* ourSource)
    : fOurSource(ourSource) {}

RawVideoBufferedPacket::~RawVideoBufferedPacket() {}

void RawVideoBufferedPacket::getNextEnclosedFrameParameters(
        unsigned char*& /*framePtr*/,
        unsigned dataSize,
        unsigned& frameSize,
        unsigned& frameDurationInMicroseconds) {
    frameDurationInMicroseconds = 0;  // because all lines within the same
                                      // packet are from the same frame

    if (fOurSource->fNextLine >= fOurSource->fNumLines) {
        fOurSource->envir()
                << "RawVideoBufferedPacket::nextEnclosedFrameParameters("
                << dataSize << "): data error (" << fOurSource->fNextLine
                << " >= " << fOurSource->fNumLines << ")!\n";
        frameSize = dataSize;
        return;
    }

    frameSize = fOurSource->fLineHeaders[fOurSource->fNextLine++].length;
}

BufferedPacket* RawVideoBufferedPacketFactory ::createNewPacket(
        MultiFramedRTPSource* ourSource) {
    return new RawVideoBufferedPacket((RawVideoRTPSource*)ourSource);
}
