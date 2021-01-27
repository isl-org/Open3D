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
// Vorbis Audio RTP Sources
// Implementation

#include "VorbisAudioRTPSource.hh"

#include "Base64.hh"

////////// VorbisBufferedPacket and VorbisBufferedPacketFactory //////////

class VorbisBufferedPacket : public BufferedPacket {
public:
    VorbisBufferedPacket();
    virtual ~VorbisBufferedPacket();

private:  // redefined virtual functions
    virtual unsigned nextEnclosedFrameSize(unsigned char*& framePtr,
                                           unsigned dataSize);
};

class VorbisBufferedPacketFactory : public BufferedPacketFactory {
private:  // redefined virtual functions
    virtual BufferedPacket* createNewPacket(MultiFramedRTPSource* ourSource);
};

///////// VorbisAudioRTPSource implementation ////////

VorbisAudioRTPSource* VorbisAudioRTPSource::createNew(
        UsageEnvironment& env,
        Groupsock* RTPgs,
        unsigned char rtpPayloadFormat,
        unsigned rtpTimestampFrequency) {
    return new VorbisAudioRTPSource(env, RTPgs, rtpPayloadFormat,
                                    rtpTimestampFrequency);
}

VorbisAudioRTPSource ::VorbisAudioRTPSource(UsageEnvironment& env,
                                            Groupsock* RTPgs,
                                            unsigned char rtpPayloadFormat,
                                            unsigned rtpTimestampFrequency)
    : MultiFramedRTPSource(env,
                           RTPgs,
                           rtpPayloadFormat,
                           rtpTimestampFrequency,
                           new VorbisBufferedPacketFactory),
      fCurPacketIdent(0) {}

VorbisAudioRTPSource::~VorbisAudioRTPSource() {}

Boolean VorbisAudioRTPSource ::processSpecialHeader(
        BufferedPacket* packet, unsigned& resultSpecialHeaderSize) {
    unsigned char* headerStart = packet->data();
    unsigned packetSize = packet->dataSize();

    resultSpecialHeaderSize = 4;
    if (packetSize < resultSpecialHeaderSize)
        return False;  // packet was too small

    // The first 3 bytes of the header are the "Ident" field:
    fCurPacketIdent =
            (headerStart[0] << 16) | (headerStart[1] << 8) | headerStart[2];

    // The 4th byte is F|VDT|numPkts.
    // Reject any packet with VDT == 3:
    if ((headerStart[3] & 0x30) == 0x30) return False;

    u_int8_t F = headerStart[3] >> 6;
    fCurrentPacketBeginsFrame = F <= 1;  // "Not Fragmented" or "Start Fragment"
    fCurrentPacketCompletesFrame =
            F == 0 || F == 3;  // "Not Fragmented" or "End Fragment"

    return True;
}

char const* VorbisAudioRTPSource::MIMEtype() const { return "audio/VORBIS"; }

////////// VorbisBufferedPacket and VorbisBufferedPacketFactory implementation
/////////////

VorbisBufferedPacket::VorbisBufferedPacket() {}

VorbisBufferedPacket::~VorbisBufferedPacket() {}

unsigned VorbisBufferedPacket ::nextEnclosedFrameSize(unsigned char*& framePtr,
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

BufferedPacket* VorbisBufferedPacketFactory ::createNewPacket(
        MultiFramedRTPSource* /*ourSource*/) {
    return new VorbisBufferedPacket();
}

////////// parseVorbisOrTheoraConfigStr() implementation //////////

#define ADVANCE(n)  \
    do {            \
        p += (n);   \
        rem -= (n); \
    } while (0)
#define GET_ENCODED_VAL(n)                 \
    do {                                   \
        u_int8_t byte;                     \
        n = 0;                             \
        do {                               \
            if (rem == 0) break;           \
            byte = *p;                     \
            n = (n * 128) + (byte & 0x7F); \
            ADVANCE(1);                    \
        } while (byte & 0x80);             \
    } while (0);                           \
    if (rem == 0) break

void parseVorbisOrTheoraConfigStr(char const* configStr,
                                  u_int8_t*& identificationHdr,
                                  unsigned& identificationHdrSize,
                                  u_int8_t*& commentHdr,
                                  unsigned& commentHdrSize,
                                  u_int8_t*& setupHdr,
                                  unsigned& setupHdrSize,
                                  u_int32_t& identField) {
    identificationHdr = commentHdr = setupHdr =
            NULL;  // default values, if an error occur
    identificationHdrSize = commentHdrSize = setupHdrSize = 0;  // ditto
    identField = 0;                                             // ditto

    // Begin by Base64-decoding the configuration string:
    unsigned configDataSize;
    u_int8_t* configData = base64Decode(configStr, configDataSize);
    u_int8_t* p = configData;
    unsigned rem = configDataSize;

    do {
        if (rem < 4) break;
        u_int32_t numPackedHeaders =
                (p[0] << 24) | (p[1] << 16) | (p[2] << 8) | p[3];
        ADVANCE(4);
        if (numPackedHeaders == 0) break;

        // Use the first 'packed header' only.
        if (rem < 3) break;
        identField = (p[0] << 16) | (p[1] << 8) | p[2];
        ADVANCE(3);

        if (rem < 2) break;
        u_int16_t length = (p[0] << 8) | p[1];
        ADVANCE(2);

        unsigned numHeaders;
        GET_ENCODED_VAL(numHeaders);

        Boolean success = False;
        for (unsigned i = 0; i < numHeaders + 1 && i < 3; ++i) {
            success = False;
            unsigned headerSize;
            if (i < numHeaders) {
                // The header size is encoded:
                GET_ENCODED_VAL(headerSize);
                if (headerSize > length) break;
                length -= headerSize;
            } else {
                // The last header is implicit:
                headerSize = length;
            }

            // Allocate space for the header bytes; we'll fill it in later
            if (i == 0) {
                identificationHdrSize = headerSize;
                identificationHdr = new u_int8_t[identificationHdrSize];
            } else if (i == 1) {
                commentHdrSize = headerSize;
                commentHdr = new u_int8_t[commentHdrSize];
            } else {  // i == 2
                setupHdrSize = headerSize;
                setupHdr = new u_int8_t[setupHdrSize];
            }

            success = True;
        }
        if (!success) break;

        // Copy the remaining config bytes into the appropriate 'header'
        // buffers:
        if (identificationHdr != NULL) {
            memmove(identificationHdr, p, identificationHdrSize);
            ADVANCE(identificationHdrSize);
            if (commentHdr != NULL) {
                memmove(commentHdr, p, commentHdrSize);
                ADVANCE(commentHdrSize);
                if (setupHdr != NULL) {
                    memmove(setupHdr, p, setupHdrSize);
                    ADVANCE(setupHdrSize);
                }
            }
        }
    } while (0);

    delete[] configData;
}
