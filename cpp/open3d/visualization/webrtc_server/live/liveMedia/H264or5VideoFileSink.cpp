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
// H.264 or H.265 Video File sinks
// Implementation

#include "H264or5VideoFileSink.hh"

#include "H264VideoRTPSource.hh"  // for "parseSPropParameterSets()"

////////// H264or5VideoFileSink //////////

H264or5VideoFileSink ::H264or5VideoFileSink(UsageEnvironment& env,
                                            FILE* fid,
                                            unsigned bufferSize,
                                            char const* perFrameFileNamePrefix,
                                            char const* sPropParameterSetsStr1,
                                            char const* sPropParameterSetsStr2,
                                            char const* sPropParameterSetsStr3)
    : FileSink(env, fid, bufferSize, perFrameFileNamePrefix),
      fHaveWrittenFirstFrame(False) {
    fSPropParameterSetsStr[0] = strDup(sPropParameterSetsStr1);
    fSPropParameterSetsStr[1] = strDup(sPropParameterSetsStr2);
    fSPropParameterSetsStr[2] = strDup(sPropParameterSetsStr3);
}

H264or5VideoFileSink::~H264or5VideoFileSink() {
    for (unsigned j = 0; j < 3; ++j) delete[](char*) fSPropParameterSetsStr[j];
}

void H264or5VideoFileSink::afterGettingFrame(unsigned frameSize,
                                             unsigned numTruncatedBytes,
                                             struct timeval presentationTime) {
    unsigned char const start_code[4] = {0x00, 0x00, 0x00, 0x01};

    if (!fHaveWrittenFirstFrame) {
        // If we have NAL units encoded in "sprop parameter strings", prepend
        // these to the file:
        for (unsigned j = 0; j < 3; ++j) {
            unsigned numSPropRecords;
            SPropRecord* sPropRecords = parseSPropParameterSets(
                    fSPropParameterSetsStr[j], numSPropRecords);
            for (unsigned i = 0; i < numSPropRecords; ++i) {
                if (sPropRecords[i].sPropLength > 0)
                    addData(start_code, 4, presentationTime);
                addData(sPropRecords[i].sPropBytes, sPropRecords[i].sPropLength,
                        presentationTime);
            }
            delete[] sPropRecords;
        }
        fHaveWrittenFirstFrame = True;  // for next time
    }

    // Write the input data to the file, with the start code in front:
    addData(start_code, 4, presentationTime);

    // Call the parent class to complete the normal file write with the input
    // data:
    FileSink::afterGettingFrame(frameSize, numTruncatedBytes, presentationTime);
}
