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
// RTP sink for Vorbis audio
// Implementation

#include "VorbisAudioRTPSink.hh"

#include "Base64.hh"
#include "VorbisAudioRTPSource.hh"  // for parseVorbisOrTheoraConfigStr()

VorbisAudioRTPSink* VorbisAudioRTPSink ::createNew(
        UsageEnvironment& env,
        Groupsock* RTPgs,
        u_int8_t rtpPayloadFormat,
        u_int32_t rtpTimestampFrequency,
        unsigned numChannels,
        u_int8_t* identificationHeader,
        unsigned identificationHeaderSize,
        u_int8_t* commentHeader,
        unsigned commentHeaderSize,
        u_int8_t* setupHeader,
        unsigned setupHeaderSize,
        u_int32_t identField) {
    return new VorbisAudioRTPSink(
            env, RTPgs, rtpPayloadFormat, rtpTimestampFrequency, numChannels,
            identificationHeader, identificationHeaderSize, commentHeader,
            commentHeaderSize, setupHeader, setupHeaderSize, identField);
}

VorbisAudioRTPSink* VorbisAudioRTPSink ::createNew(
        UsageEnvironment& env,
        Groupsock* RTPgs,
        u_int8_t rtpPayloadFormat,
        u_int32_t rtpTimestampFrequency,
        unsigned numChannels,
        char const* configStr) {
    // Begin by decoding and unpacking the configuration string:
    u_int8_t* identificationHeader;
    unsigned identificationHeaderSize;
    u_int8_t* commentHeader;
    unsigned commentHeaderSize;
    u_int8_t* setupHeader;
    unsigned setupHeaderSize;
    u_int32_t identField;

    parseVorbisOrTheoraConfigStr(configStr, identificationHeader,
                                 identificationHeaderSize, commentHeader,
                                 commentHeaderSize, setupHeader,
                                 setupHeaderSize, identField);

    VorbisAudioRTPSink* resultSink = new VorbisAudioRTPSink(
            env, RTPgs, rtpPayloadFormat, rtpTimestampFrequency, numChannels,
            identificationHeader, identificationHeaderSize, commentHeader,
            commentHeaderSize, setupHeader, setupHeaderSize, identField);
    delete[] identificationHeader;
    delete[] commentHeader;
    delete[] setupHeader;

    return resultSink;
}

VorbisAudioRTPSink ::VorbisAudioRTPSink(UsageEnvironment& env,
                                        Groupsock* RTPgs,
                                        u_int8_t rtpPayloadFormat,
                                        u_int32_t rtpTimestampFrequency,
                                        unsigned numChannels,
                                        u_int8_t* identificationHeader,
                                        unsigned identificationHeaderSize,
                                        u_int8_t* commentHeader,
                                        unsigned commentHeaderSize,
                                        u_int8_t* setupHeader,
                                        unsigned setupHeaderSize,
                                        u_int32_t identField)
    : AudioRTPSink(env,
                   RTPgs,
                   rtpPayloadFormat,
                   rtpTimestampFrequency,
                   "VORBIS",
                   numChannels),
      fIdent(identField),
      fFmtpSDPLine(NULL) {
    if (identificationHeaderSize >= 28) {
        // Get the 'bitrate' values from this header, and use them to set our
        // estimated bitrate:
        u_int32_t val;
        u_int8_t* p;

        p = &identificationHeader[16];
        val = ((p[3] * 256 + p[2]) * 256 + p[1]) * 256 +
              p[0];  // i.e., little-endian
        int bitrate_maximum = (int)val;
        if (bitrate_maximum < 0) bitrate_maximum = 0;

        p = &identificationHeader[20];
        val = ((p[3] * 256 + p[2]) * 256 + p[1]) * 256 +
              p[0];  // i.e., little-endian
        int bitrate_nominal = (int)val;
        if (bitrate_nominal < 0) bitrate_nominal = 0;

        p = &identificationHeader[24];
        val = ((p[3] * 256 + p[2]) * 256 + p[1]) * 256 +
              p[0];  // i.e., little-endian
        int bitrate_minimum = (int)val;
        if (bitrate_minimum < 0) bitrate_minimum = 0;

        int bitrate =
                bitrate_nominal > 0
                        ? bitrate_nominal
                        : bitrate_maximum > 0
                                  ? bitrate_maximum
                                  : bitrate_minimum > 0 ? bitrate_minimum : 0;
        if (bitrate > 0) estimatedBitrate() = ((unsigned)bitrate) / 1000;
    }

    // Generate a 'config' string from the supplied configuration headers:
    char* base64PackedHeaders = generateVorbisOrTheoraConfigStr(
            identificationHeader, identificationHeaderSize, commentHeader,
            commentHeaderSize, setupHeader, setupHeaderSize, identField);
    if (base64PackedHeaders == NULL) return;

    // Then use this 'config' string to construct our "a=fmtp:" SDP line:
    unsigned fmtpSDPLineMaxSize =
            50 + strlen(base64PackedHeaders);  // 50 => more than enough space
    fFmtpSDPLine = new char[fmtpSDPLineMaxSize];
    sprintf(fFmtpSDPLine, "a=fmtp:%d configuration=%s\r\n", rtpPayloadType(),
            base64PackedHeaders);
    delete[] base64PackedHeaders;
}

VorbisAudioRTPSink::~VorbisAudioRTPSink() { delete[] fFmtpSDPLine; }

char const* VorbisAudioRTPSink::auxSDPLine() { return fFmtpSDPLine; }

void VorbisAudioRTPSink ::doSpecialFrameHandling(
        unsigned fragmentationOffset,
        unsigned char* frameStart,
        unsigned numBytesInFrame,
        struct timeval framePresentationTime,
        unsigned numRemainingBytes) {
    // Set the 4-byte "payload header", as defined in RFC 5215, section 2.2:
    u_int8_t header[4];

    // The first three bytes of the header are our "Ident":
    header[0] = fIdent >> 16;
    header[1] = fIdent >> 8;
    header[2] = fIdent;

    // The final byte contains the "F", "VDT", and "numPkts" fields:
    u_int8_t F;  // Fragment type
    if (numRemainingBytes > 0) {
        if (fragmentationOffset > 0) {
            F = 2 << 6;  // continuation fragment
        } else {
            F = 1 << 6;  // start fragment
        }
    } else {
        if (fragmentationOffset > 0) {
            F = 3 << 6;  // end fragment
        } else {
            F = 0 << 6;  // not fragmented
        }
    }
    u_int8_t const VDT =
            0 << 4;  // Vorbis Data Type (always a "Raw Vorbis payload")
    u_int8_t numPkts = F == 0 ? (numFramesUsedSoFar() + 1)
                              : 0;  // set to 0 when we're a fragment
    header[3] = F | VDT | numPkts;

    setSpecialHeaderBytes(header, sizeof header);

    // There's also a 2-byte 'frame-specific' header: The length of the Vorbis
    // data:
    u_int8_t frameSpecificHeader[2];
    frameSpecificHeader[0] = numBytesInFrame >> 8;
    frameSpecificHeader[1] = numBytesInFrame;
    setFrameSpecificHeaderBytes(frameSpecificHeader, 2);

    // Important: Also call our base class's doSpecialFrameHandling(),
    // to set the packet's timestamp:
    MultiFramedRTPSink::doSpecialFrameHandling(
            fragmentationOffset, frameStart, numBytesInFrame,
            framePresentationTime, numRemainingBytes);
}

Boolean VorbisAudioRTPSink::frameCanAppearAfterPacketStart(
        unsigned char const* /*frameStart*/,
        unsigned /*numBytesInFrame*/) const {
    // We allow more than one frame to be packed into an outgoing RTP packet,
    // but no more than 15:
    return numFramesUsedSoFar() <= 15;
}

unsigned VorbisAudioRTPSink::specialHeaderSize() const { return 4; }

unsigned VorbisAudioRTPSink::frameSpecificHeaderSize() const { return 2; }

////////// generateVorbisOrTheoraConfigStr() implementation //////////

char* generateVorbisOrTheoraConfigStr(u_int8_t* identificationHeader,
                                      unsigned identificationHeaderSize,
                                      u_int8_t* commentHeader,
                                      unsigned commentHeaderSize,
                                      u_int8_t* setupHeader,
                                      unsigned setupHeaderSize,
                                      u_int32_t identField) {
    // First, count how many headers (<=3) are included, and how many bytes will
    // be used to encode these headers' sizes:
    unsigned numHeaders = 0;
    unsigned sizeSize[2];  // The number of bytes used to encode the lengths of
                           // the first two headers (but not the length of the
                           // 3rd)
    sizeSize[0] = sizeSize[1] = 0;
    if (identificationHeaderSize > 0) {
        sizeSize[numHeaders++] =
                identificationHeaderSize < 128
                        ? 1
                        : identificationHeaderSize < 16384 ? 2 : 3;
    }
    if (commentHeaderSize > 0) {
        sizeSize[numHeaders++] =
                commentHeaderSize < 128 ? 1 : commentHeaderSize < 16384 ? 2 : 3;
    }
    if (setupHeaderSize > 0) {
        ++numHeaders;
    } else {
        sizeSize[1] = 0;  // We have at most two headers, so the second one's
                          // length isn't encoded
    }
    if (numHeaders == 0)
        return NULL;  // With no headers, we can't set up a configuration
    if (numHeaders == 1)
        sizeSize[0] = 0;  // With only one header, its length isn't encoded

    // Then figure out the size of the packed configuration headers, and
    // allocate space for this:
    unsigned length =
            identificationHeaderSize + commentHeaderSize + setupHeaderSize;
    // The "length" field in the packed headers
    if (length > (unsigned)0xFFFF)
        return NULL;  // too big for a 16-bit field; we can't handle this
    unsigned packedHeadersSize =
            4    // "Number of packed headers" field
            + 3  // "ident" field
            + 2  // "length" field
            + 1  // "n. of headers" field
            + sizeSize[0] +
            sizeSize[1]  // "length1" and "length2" (if present) fields
            + length;
    u_int8_t* packedHeaders = new u_int8_t[packedHeadersSize];
    if (packedHeaders == NULL) return NULL;

    // Fill in the 'packed headers':
    u_int8_t* p = packedHeaders;
    *p++ = 0;
    *p++ = 0;
    *p++ = 0;
    *p++ = 1;  // "Number of packed headers": 1
    *p++ = identField >> 16;
    *p++ = identField >> 8;
    *p++ = identField;  // "Ident" (24 bits)
    *p++ = length >> 8;
    *p++ = length;          // "length" (16 bits)
    *p++ = numHeaders - 1;  // "n. of headers"
    if (numHeaders > 1) {
        // Fill in the "length1" header:
        unsigned length1 = identificationHeaderSize > 0
                                   ? identificationHeaderSize
                                   : commentHeaderSize;
        if (length1 >= 16384) {
            *p++ = 0x80;  // flag, but no more, because we know length1 <= 32767
        }
        if (length1 >= 128) {
            *p++ = 0x80 |
                   ((length1 & 0x3F80) >> 7);  // flag + the second 7 bits
        }
        *p++ = length1 & 0x7F;  // the low 7 bits

        if (numHeaders > 2) {  // numHeaders == 3
            // Fill in the "length2" header (for the 'Comment' header):
            unsigned length2 = commentHeaderSize;
            if (length2 >= 16384) {
                *p++ = 0x80;  // flag, but no more, because we know length2 <=
                              // 32767
            }
            if (length2 >= 128) {
                *p++ = 0x80 |
                       ((length2 & 0x3F80) >> 7);  // flag + the second 7 bits
            }
            *p++ = length2 & 0x7F;  // the low 7 bits
        }
    }
    // Copy each header:
    if (identificationHeader != NULL)
        memmove(p, identificationHeader, identificationHeaderSize);
    p += identificationHeaderSize;
    if (commentHeader != NULL) memmove(p, commentHeader, commentHeaderSize);
    p += commentHeaderSize;
    if (setupHeader != NULL) memmove(p, setupHeader, setupHeaderSize);

    // Having set up the 'packed configuration headers', Base-64-encode this,
    // for our result:
    char* base64PackedHeaders =
            base64Encode((char const*)packedHeaders, packedHeadersSize);
    delete[] packedHeaders;

    return base64PackedHeaders;
}
