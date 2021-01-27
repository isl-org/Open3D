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
// RTP sink for Raw video
// Implementation

#include "RawVideoRTPSink.hh"

RawVideoRTPSink* RawVideoRTPSink ::createNew(UsageEnvironment& env,
                                             Groupsock* RTPgs,
                                             u_int8_t rtpPayloadFormat,
                                             unsigned height,
                                             unsigned width,
                                             unsigned depth,
                                             char const* sampling,
                                             char const* colorimetry) {
    return new RawVideoRTPSink(env, RTPgs, rtpPayloadFormat, height, width,
                               depth, sampling, colorimetry);
}

RawVideoRTPSink ::RawVideoRTPSink(UsageEnvironment& env,
                                  Groupsock* RTPgs,
                                  u_int8_t rtpPayloadFormat,
                                  unsigned height,
                                  unsigned width,
                                  unsigned depth,
                                  char const* sampling,
                                  char const* colorimetry)
    : VideoRTPSink(env, RTPgs, rtpPayloadFormat, 90000, "RAW"),
      fFmtpSDPLine(NULL),
      fSampling(NULL),
      fWidth(width),
      fHeight(height),
      fDepth(depth),
      fColorimetry(NULL),
      fLineindex(0) {
    // Then use this 'config' string to construct our "a=fmtp:" SDP line:
    unsigned fmtpSDPLineMaxSize = 200;  // 200 => more than enough space
    fFmtpSDPLine = new char[fmtpSDPLineMaxSize];
    sprintf(fFmtpSDPLine,
            "a=fmtp:%d "
            "sampling=%s;width=%u;height=%u;depth=%u;colorimetry=%s\r\n",
            rtpPayloadType(), sampling, width, height, depth, colorimetry);

    // Set parameters
    fSampling = strDup(sampling);
    fColorimetry = strDup(colorimetry);
    setFrameParameters();
}

RawVideoRTPSink::~RawVideoRTPSink() {
    delete[] fFmtpSDPLine;
    delete[] fSampling;
    delete[] fColorimetry;
}

char const* RawVideoRTPSink::auxSDPLine() { return fFmtpSDPLine; }

void RawVideoRTPSink ::doSpecialFrameHandling(
        unsigned fragmentationOffset,
        unsigned char* frameStart,
        unsigned numBytesInFrame,
        struct timeval framePresentationTime,
        unsigned numRemainingBytes) {
    unsigned* lengths = NULL;
    unsigned* offsets = NULL;
    unsigned nbLines = getNbLineInPacket(fragmentationOffset, lengths, offsets);
    unsigned specialHeaderSize = 2 + (6 * nbLines);
    u_int8_t* specialHeader = new u_int8_t[specialHeaderSize];

    // Extended Sequence Number (not used)
    specialHeader[0] = 0;
    specialHeader[1] = 0;

    for (unsigned i = 0; i < nbLines; i++) {
        // detection of new line incrementation
        if ((offsets[i] == 0) && fragmentationOffset != 0) {
            fLineindex = fLineindex + fFrameParameters.scanLineIterationStep;
        }

        // Set length
        specialHeader[2 + (i * 6) + 0] = lengths[i] >> 8;
        specialHeader[2 + (i * 6) + 1] = (u_int8_t)lengths[i];

        // Field Identification (false for us)
        bool fieldIdent = false;

        // Set line index
        specialHeader[2 + (i * 6) + 2] =
                ((fLineindex >> 8) & 0x7F) | (fieldIdent << 7);
        specialHeader[2 + (i * 6) + 3] = (u_int8_t)fLineindex;

        // Set Continuation bit
        bool continuationBit = (i < nbLines - 1) ? true : false;

        // Set offset
        specialHeader[2 + (i * 6) + 4] =
                ((offsets[i] >> 8) & 0x7F) | (continuationBit << 7);
        specialHeader[2 + (i * 6) + 5] = (u_int8_t)offsets[i];
    }

    setSpecialHeaderBytes(specialHeader, specialHeaderSize);

    if (numRemainingBytes == 0) {
        // This packet contains the last (or only) fragment of the frame.
        // Set the RTP 'M' ('marker') bit:
        setMarkerBit();
        // Reset line index
        fLineindex = 0;
    }

    // Also set the RTP timestamp:
    setTimestamp(framePresentationTime);

    delete[] specialHeader;
    delete[] lengths;
    delete[] offsets;
}

Boolean RawVideoRTPSink::frameCanAppearAfterPacketStart(
        unsigned char const* /*frameStart*/,
        unsigned /*numBytesInFrame*/) const {
    // Only one frame per packet:
    return False;
}

unsigned RawVideoRTPSink::specialHeaderSize() const {
    unsigned* lengths = NULL;
    unsigned* offsets = NULL;
    unsigned nbLines =
            getNbLineInPacket(curFragmentationOffset(), lengths, offsets);
    delete[] lengths;
    delete[] offsets;
    return 2 + (6 * nbLines);
}

unsigned RawVideoRTPSink::getNbLineInPacket(unsigned fragOffset,
                                            unsigned*& lengths,
                                            unsigned*& offsets) const {
    unsigned rtpHeaderSize = 12;
    unsigned specialHeaderSize = 2;  // Extended Sequence Nb
    unsigned packetMaxSize = ourMaxPacketSize();
    unsigned nbLines = 0;
    unsigned remainingSizeInPacket;

    if (fragOffset >= fFrameParameters.frameSize) {
        envir() << "RawVideoRTPSink::getNbLineInPacket(): bad fragOffset "
                << fragOffset << "\n";
        return 0;
    }
    unsigned lengthArray[100] = {0};
    unsigned offsetArray[100] = {0};
    unsigned curDataTotalLength = 0;
    unsigned lineOffset = (fragOffset % fFrameParameters.scanLineSize);

    unsigned remainingLineSize = fFrameParameters.scanLineSize -
                                 (fragOffset % fFrameParameters.scanLineSize);
    while (1) {
        if (packetMaxSize - specialHeaderSize - rtpHeaderSize - 6 <=
            curDataTotalLength) {
            break;  // packet sanity check
        }

        // add one line
        nbLines++;
        specialHeaderSize += 6;

        remainingSizeInPacket = packetMaxSize - specialHeaderSize -
                                rtpHeaderSize - curDataTotalLength;
        remainingSizeInPacket -=
                remainingSizeInPacket %
                fFrameParameters.pGroupSize;  // use only multiple of pgroup
        lengthArray[nbLines - 1] = remainingLineSize < remainingSizeInPacket
                                           ? remainingLineSize
                                           : remainingSizeInPacket;
        offsetArray[nbLines - 1] = lineOffset *
                                   fFrameParameters.scanLineIterationStep /
                                   fFrameParameters.pGroupSize;
        if (remainingLineSize >= remainingSizeInPacket) {
            break;  // packet full
        }

        remainingLineSize = fFrameParameters.scanLineSize;
        curDataTotalLength += lengthArray[nbLines - 1];
        lineOffset = 0;

        if (fragOffset + curDataTotalLength >= fFrameParameters.frameSize) {
            break;  // end of the frame.
        }
    }

    lengths = new unsigned[nbLines];
    offsets = new unsigned[nbLines];
    for (unsigned i = 0; i < nbLines; i++) {
        lengths[i] = lengthArray[i];
        offsets[i] = offsetArray[i];
    }
    return nbLines;
}

unsigned RawVideoRTPSink::computeOverflowForNewFrame(
        unsigned newFrameSize) const {
    unsigned initialOverflow =
            MultiFramedRTPSink::computeOverflowForNewFrame(newFrameSize);

    // Adjust (increase) this overflow to be a multiple of the pgroup value
    unsigned numFrameBytesUsed = newFrameSize - initialOverflow;
    initialOverflow += numFrameBytesUsed % fFrameParameters.pGroupSize;

    return initialOverflow;
}

void RawVideoRTPSink::setFrameParameters() {
    fFrameParameters.scanLineIterationStep = 1;
    if ((strncmp("RGB", fSampling, strlen(fSampling)) == 0) ||
        (strncmp("BGR", fSampling, strlen(fSampling)) == 0)) {
        switch (fDepth) {
            case 8:
                fFrameParameters.pGroupSize = 3;
                fFrameParameters.nbOfPixelInPGroup = 1;
                break;
            case 10:
                fFrameParameters.pGroupSize = 15;
                fFrameParameters.nbOfPixelInPGroup = 4;
                break;
            case 12:
                fFrameParameters.pGroupSize = 9;
                fFrameParameters.nbOfPixelInPGroup = 2;
                break;
            case 16:
                fFrameParameters.pGroupSize = 6;
                fFrameParameters.nbOfPixelInPGroup = 1;
                break;
            default:
                break;
        }
    } else if ((strncmp("RGBA", fSampling, strlen(fSampling)) == 0) ||
               (strncmp("BGRA", fSampling, strlen(fSampling)) == 0)) {
        switch (fDepth) {
            case 8:
                fFrameParameters.pGroupSize = 4;
                break;
            case 10:
                fFrameParameters.pGroupSize = 5;
                break;
            case 12:
                fFrameParameters.pGroupSize = 6;
                break;
            case 16:
                fFrameParameters.pGroupSize = 8;
                break;
            default:
                break;
        }
        fFrameParameters.nbOfPixelInPGroup = 1;
    } else if (strncmp("YCbCr-4:4:4", fSampling, strlen(fSampling)) == 0) {
        switch (fDepth) {
            case 8:
                fFrameParameters.pGroupSize = 3;
                fFrameParameters.nbOfPixelInPGroup = 1;
                break;
            case 10:
                fFrameParameters.pGroupSize = 15;
                fFrameParameters.nbOfPixelInPGroup = 4;
                break;
            case 12:
                fFrameParameters.pGroupSize = 9;
                fFrameParameters.nbOfPixelInPGroup = 2;
                break;
            case 16:
                fFrameParameters.pGroupSize = 6;
                fFrameParameters.nbOfPixelInPGroup = 1;
                break;
            default:
                break;
        }
    } else if (strncmp("YCbCr-4:2:2", fSampling, strlen(fSampling)) == 0) {
        switch (fDepth) {
            case 8:
                fFrameParameters.pGroupSize = 4;
                break;
            case 10:
                fFrameParameters.pGroupSize = 5;
                break;
            case 12:
                fFrameParameters.pGroupSize = 6;
                break;
            case 16:
                fFrameParameters.pGroupSize = 8;
                break;
            default:
                break;
        }
        fFrameParameters.nbOfPixelInPGroup = 2;
    } else if (strncmp("YCbCr-4:1:1", fSampling, strlen(fSampling)) == 0) {
        switch (fDepth) {
            case 8:
                fFrameParameters.pGroupSize = 6;
                break;
            case 10:
                fFrameParameters.pGroupSize = 15;
                break;
            case 12:
                fFrameParameters.pGroupSize = 9;
                break;
            case 16:
                fFrameParameters.pGroupSize = 12;
                break;
            default:
                break;
        }
        fFrameParameters.nbOfPixelInPGroup = 4;
    } else if (strncmp("YCbCr-4:2:0", fSampling, strlen(fSampling)) == 0) {
        switch (fDepth) {
            case 8:
                fFrameParameters.pGroupSize = 6;
                break;
            case 10:
                fFrameParameters.pGroupSize = 15;
                break;
            case 12:
                fFrameParameters.pGroupSize = 9;
                break;
            case 16:
                fFrameParameters.pGroupSize = 12;
                break;
            default:
                break;
        }
        fFrameParameters.nbOfPixelInPGroup = 4;
        fFrameParameters.scanLineIterationStep = 2;
    }
    fFrameParameters.frameSize = fHeight * fWidth *
                                 fFrameParameters.pGroupSize /
                                 fFrameParameters.nbOfPixelInPGroup;
    fFrameParameters.scanLineSize = fWidth * fFrameParameters.pGroupSize /
                                    fFrameParameters.nbOfPixelInPGroup *
                                    fFrameParameters.scanLineIterationStep;
}
