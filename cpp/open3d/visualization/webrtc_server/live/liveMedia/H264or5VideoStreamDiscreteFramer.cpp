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
// A simplified version of "H264or5VideoStreamFramer" that takes only complete,
// discrete frames (rather than an arbitrary byte stream) as input.
// This avoids the parsing and data copying overhead of the full
// "H264or5VideoStreamFramer".
// Implementation

#include "H264or5VideoStreamDiscreteFramer.hh"

H264or5VideoStreamDiscreteFramer ::H264or5VideoStreamDiscreteFramer(
        int hNumber,
        UsageEnvironment& env,
        FramedSource* inputSource,
        Boolean includeStartCodeInOutput,
        Boolean insertAccessUnitDelimiters)
    : H264or5VideoStreamFramer(hNumber,
                               env,
                               inputSource,
                               False /*don't create a parser*/,
                               includeStartCodeInOutput,
                               insertAccessUnitDelimiters) {}

H264or5VideoStreamDiscreteFramer::~H264or5VideoStreamDiscreteFramer() {}

void H264or5VideoStreamDiscreteFramer::doGetNextFrame() {
    if (fIncludeStartCodeInOutput) {
        // Prepend a 4-byte 'start code' (0x00000001) to the output:
        if (fMaxSize < 4) {  // there's no space
            fNumTruncatedBytes = 4 - fMaxSize;
            handleClosure();
            return;
        }
        *fTo++ = 0x00;
        *fTo++ = 0x00;
        *fTo++ = 0x00;
        *fTo++ = 0x01;
        fMaxSize -= 4;
    }

    if (fInsertAccessUnitDelimiters && pictureEndMarker()) {
        // Deliver an "access_unit_delimiter" NAL unit instead:
        unsigned const audNALSize = fHNumber == 264 ? 2 : 3;

        if (audNALSize > fMaxSize) {  // there's no space
            fNumTruncatedBytes = audNALSize - fMaxSize;
            handleClosure();
            return;
        }

        if (fHNumber == 264) {
            *fTo++ = 9;        // "Access unit delimiter" nal_unit_type
            *fTo++ = 0xF0;     // "primary_pic_type" (7); "rbsp_trailing_bits()"
        } else {               // H.265
            *fTo++ = 35 << 1;  // "Access unit delimiter" nal_unit_type
            *fTo++ = 0;  // "nuh_layer_id" (0); "nuh_temporal_id_plus1" (0) (Is
                         // this correct??)
            *fTo++ = 0x50;  // "pic_type" (2); "rbsp_trailing_bits()" (Is this
                            // correct??)
        }

        fFrameSize = (fIncludeStartCodeInOutput ? 4 : 0) + audNALSize;
        pictureEndMarker() = False;  // for next time
        afterGetting(this);  // complete delivery to the downstream object
    } else {
        // Normal case:
        // Arrange to read data (which should be a complete H.264 or H.265 NAL
        // unit) from our data source, directly into the client's input buffer.
        // After reading this, we'll do some parsing on the frame.
        fInputSource->getNextFrame(fTo, fMaxSize, afterGettingFrame, this,
                                   FramedSource::handleClosure, this);
    }
}

void H264or5VideoStreamDiscreteFramer ::afterGettingFrame(
        void* clientData,
        unsigned frameSize,
        unsigned numTruncatedBytes,
        struct timeval presentationTime,
        unsigned durationInMicroseconds) {
    H264or5VideoStreamDiscreteFramer* source =
            (H264or5VideoStreamDiscreteFramer*)clientData;
    source->afterGettingFrame1(frameSize, numTruncatedBytes, presentationTime,
                               durationInMicroseconds);
}

void H264or5VideoStreamDiscreteFramer ::afterGettingFrame1(
        unsigned frameSize,
        unsigned numTruncatedBytes,
        struct timeval presentationTime,
        unsigned durationInMicroseconds) {
    // Get the "nal_unit_type", to see if this NAL unit is one that we want to
    // save a copy of:
    u_int8_t nal_unit_type;
    if (fHNumber == 264 && frameSize >= 1) {
        nal_unit_type = fTo[0] & 0x1F;
    } else if (fHNumber == 265 && frameSize >= 2) {
        nal_unit_type = (fTo[0] & 0x7E) >> 1;
    } else {
        // This is too short to be a valid NAL unit, so just assume a bogus
        // nal_unit_type
        nal_unit_type = 0xFF;
    }

    // Begin by checking for a (likely) common error: NAL units that
    // (erroneously) begin with a 0x00000001 or 0x000001 'start code'.  (Those
    // start codes should only be in byte-stream data; *not* data that consists
    // of discrete NAL units.) Once again, to be clear: The NAL units that you
    // feed to a "H264or5VideoStreamDiscreteFramer" MUST NOT include start
    // codes.
    if (frameSize >= 4 && fTo[0] == 0 && fTo[1] == 0 &&
        ((fTo[2] == 0 && fTo[3] == 1) || fTo[2] == 1)) {
        envir() << "H264or5VideoStreamDiscreteFramer error: MPEG 'start code' "
                   "seen in the input\n";
    } else if (isVPS(nal_unit_type)) {  // Video parameter set (VPS)
        saveCopyOfVPS(fTo, frameSize);
    } else if (isSPS(nal_unit_type)) {  // Sequence parameter set (SPS)
        saveCopyOfSPS(fTo, frameSize);
    } else if (isPPS(nal_unit_type)) {  // Picture parameter set (PPS)
        saveCopyOfPPS(fTo, frameSize);
    }

    fPictureEndMarker = nalUnitEndsAccessUnit(nal_unit_type);

    // Finally, complete delivery to the client:
    fFrameSize = fIncludeStartCodeInOutput ? (4 + frameSize) : frameSize;
    fNumTruncatedBytes = numTruncatedBytes;
    fPresentationTime = presentationTime;
    fDurationInMicroseconds = durationInMicroseconds;
    afterGetting(this);
}

Boolean H264or5VideoStreamDiscreteFramer::nalUnitEndsAccessUnit(
        u_int8_t nal_unit_type) {
    // Check whether this NAL unit ends the current 'access unit' (basically, a
    // video frame).
    //  Unfortunately, we can't do this reliably, because we don't yet know
    //  anything about the
    // *next* NAL unit that we'll see.  So, we guess this as best as we can, by
    // assuming that if this NAL unit is a VCL NAL unit, then it ends the
    // current 'access unit'.
    //
    // This will be wrong if you are streaming multiple 'slices' per picture. In
    // that case, you can define a subclass that reimplements this virtual
    // function to do the right thing.

    return isVCL(nal_unit_type);
}
