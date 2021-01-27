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
// H.264 Video File sinks
// Implementation

#include "H264VideoFileSink.hh"

#include "OutputFile.hh"

////////// H264VideoFileSink //////////

H264VideoFileSink ::H264VideoFileSink(UsageEnvironment& env,
                                      FILE* fid,
                                      char const* sPropParameterSetsStr,
                                      unsigned bufferSize,
                                      char const* perFrameFileNamePrefix)
    : H264or5VideoFileSink(env,
                           fid,
                           bufferSize,
                           perFrameFileNamePrefix,
                           sPropParameterSetsStr,
                           NULL,
                           NULL) {}

H264VideoFileSink::~H264VideoFileSink() {}

H264VideoFileSink* H264VideoFileSink::createNew(
        UsageEnvironment& env,
        char const* fileName,
        char const* sPropParameterSetsStr,
        unsigned bufferSize,
        Boolean oneFilePerFrame) {
    do {
        FILE* fid;
        char const* perFrameFileNamePrefix;
        if (oneFilePerFrame) {
            // Create the fid for each frame
            fid = NULL;
            perFrameFileNamePrefix = fileName;
        } else {
            // Normal case: create the fid once
            fid = OpenOutputFile(env, fileName);
            if (fid == NULL) break;
            perFrameFileNamePrefix = NULL;
        }

        return new H264VideoFileSink(env, fid, sPropParameterSetsStr,
                                     bufferSize, perFrameFileNamePrefix);
    } while (0);

    return NULL;
}
