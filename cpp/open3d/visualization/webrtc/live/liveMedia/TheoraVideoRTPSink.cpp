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
// RTP sink for Theora video
// Implementation

#include "TheoraVideoRTPSink.hh"

#include "Base64.hh"
#include "VorbisAudioRTPSink.hh"    // for generateVorbisOrTheoraConfigStr()
#include "VorbisAudioRTPSource.hh"  // for parseVorbisOrTheoraConfigStr()

TheoraVideoRTPSink* TheoraVideoRTPSink ::createNew(
        UsageEnvironment& env,
        Groupsock* RTPgs,
        u_int8_t rtpPayloadFormat,
        u_int8_t* identificationHeader,
        unsigned identificationHeaderSize,
        u_int8_t* commentHeader,
        unsigned commentHeaderSize,
        u_int8_t* setupHeader,
        unsigned setupHeaderSize,
        u_int32_t identField) {
    return new TheoraVideoRTPSink(
            env, RTPgs, rtpPayloadFormat, identificationHeader,
            identificationHeaderSize, commentHeader, commentHeaderSize,
            setupHeader, setupHeaderSize, identField);
}

TheoraVideoRTPSink* TheoraVideoRTPSink ::createNew(UsageEnvironment& env,
                                                   Groupsock* RTPgs,
                                                   u_int8_t rtpPayloadFormat,
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

    TheoraVideoRTPSink* resultSink = new TheoraVideoRTPSink(
            env, RTPgs, rtpPayloadFormat, identificationHeader,
            identificationHeaderSize, commentHeader, commentHeaderSize,
            setupHeader, setupHeaderSize, identField);
    delete[] identificationHeader;
    delete[] commentHeader;
    delete[] setupHeader;

    return resultSink;
}

TheoraVideoRTPSink ::TheoraVideoRTPSink(UsageEnvironment& env,
                                        Groupsock* RTPgs,
                                        u_int8_t rtpPayloadFormat,
                                        u_int8_t* identificationHeader,
                                        unsigned identificationHeaderSize,
                                        u_int8_t* commentHeader,
                                        unsigned commentHeaderSize,
                                        u_int8_t* setupHeader,
                                        unsigned setupHeaderSize,
                                        u_int32_t identField)
    : VideoRTPSink(env, RTPgs, rtpPayloadFormat, 90000, "THEORA"),
      fIdent(identField),
      fFmtpSDPLine(NULL) {
    static const char* pf_to_str[] = {
            "YCbCr-4:2:0",
            "Reserved",
            "YCbCr-4:2:2",
            "YCbCr-4:4:4",
    };

    unsigned width = 1280;  // default value
    unsigned height = 720;  // default value
    unsigned pf = 0;        // default value
    if (identificationHeaderSize >= 42) {
        // Parse this header to get the "width", "height", "pf" (pixel format),
        // and 'nominal bitrate' parameters:
        u_int8_t* p = identificationHeader;  // alias
        width = (p[14] << 16) | (p[15] << 8) | p[16];
        height = (p[17] << 16) | (p[18] << 8) | p[19];
        pf = (p[41] & 0x18) >> 3;
        unsigned nominalBitrate = (p[37] << 16) | (p[38] << 8) | p[39];
        if (nominalBitrate > 0) estimatedBitrate() = nominalBitrate / 1000;
    }

    // Generate a 'config' string from the supplied configuration headers:
    char* base64PackedHeaders = generateVorbisOrTheoraConfigStr(
            identificationHeader, identificationHeaderSize, commentHeader,
            commentHeaderSize, setupHeader, setupHeaderSize, identField);
    if (base64PackedHeaders == NULL) return;

    // Then use this 'config' string to construct our "a=fmtp:" SDP line:
    unsigned fmtpSDPLineMaxSize =
            200 + strlen(base64PackedHeaders);  // 200 => more than enough space
    fFmtpSDPLine = new char[fmtpSDPLineMaxSize];
    sprintf(fFmtpSDPLine,
            "a=fmtp:%d "
            "sampling=%s;width=%u;height=%u;delivery-method=out_band/"
            "rtsp;configuration=%s\r\n",
            rtpPayloadType(), pf_to_str[pf], width, height,
            base64PackedHeaders);
    delete[] base64PackedHeaders;
}

TheoraVideoRTPSink::~TheoraVideoRTPSink() { delete[] fFmtpSDPLine; }

char const* TheoraVideoRTPSink::auxSDPLine() { return fFmtpSDPLine; }

void TheoraVideoRTPSink ::doSpecialFrameHandling(
        unsigned fragmentationOffset,
        unsigned char* frameStart,
        unsigned numBytesInFrame,
        struct timeval framePresentationTime,
        unsigned numRemainingBytes) {
    // Set the 4-byte "payload header", as defined in
    // http://svn.xiph.org/trunk/theora/doc/draft-ietf-avt-rtp-theora-00.txt
    u_int8_t header[6];

    // The three bytes of the header are our "Ident":
    header[0] = fIdent >> 16;
    header[1] = fIdent >> 8;
    header[2] = fIdent;

    // The final byte contains the "F", "TDT", and "numPkts" fields:
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
    u_int8_t const TDT =
            0 << 4;  // Theora Data Type (always a "Raw Theora payload")
    u_int8_t numPkts = F == 0 ? (numFramesUsedSoFar() + 1)
                              : 0;  // set to 0 when we're a fragment
    header[3] = F | TDT | numPkts;

    // There's also a 2-byte 'frame-specific' header: The length of the
    // Theora data:
    header[4] = numBytesInFrame >> 8;
    header[5] = numBytesInFrame;
    setSpecialHeaderBytes(header, sizeof(header));

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

Boolean TheoraVideoRTPSink::frameCanAppearAfterPacketStart(
        unsigned char const* /*frameStart*/,
        unsigned /*numBytesInFrame*/) const {
    // Only one frame per packet:
    return False;
}

unsigned TheoraVideoRTPSink::specialHeaderSize() const { return 6; }
