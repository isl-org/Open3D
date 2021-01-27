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
// File sinks
// Implementation

#if (defined(__WIN32__) || defined(_WIN32)) && !defined(_WIN32_WCE)
#include <fcntl.h>
#include <io.h>
#endif
#include "FileSink.hh"
#include "GroupsockHelper.hh"
#include "OutputFile.hh"

////////// FileSink //////////

FileSink::FileSink(UsageEnvironment& env,
                   FILE* fid,
                   unsigned bufferSize,
                   char const* perFrameFileNamePrefix)
    : MediaSink(env),
      fOutFid(fid),
      fBufferSize(bufferSize),
      fSamePresentationTimeCounter(0) {
    fBuffer = new unsigned char[bufferSize];
    if (perFrameFileNamePrefix != NULL) {
        fPerFrameFileNamePrefix = strDup(perFrameFileNamePrefix);
        fPerFrameFileNameBuffer =
                new char[strlen(perFrameFileNamePrefix) + 100];
    } else {
        fPerFrameFileNamePrefix = NULL;
        fPerFrameFileNameBuffer = NULL;
    }
    fPrevPresentationTime.tv_sec = ~0;
    fPrevPresentationTime.tv_usec = 0;
}

FileSink::~FileSink() {
    delete[] fPerFrameFileNameBuffer;
    delete[] fPerFrameFileNamePrefix;
    delete[] fBuffer;
    if (fOutFid != NULL) fclose(fOutFid);
}

FileSink* FileSink::createNew(UsageEnvironment& env,
                              char const* fileName,
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

        return new FileSink(env, fid, bufferSize, perFrameFileNamePrefix);
    } while (0);

    return NULL;
}

Boolean FileSink::continuePlaying() {
    if (fSource == NULL) return False;

    fSource->getNextFrame(fBuffer, fBufferSize, afterGettingFrame, this,
                          onSourceClosure, this);

    return True;
}

void FileSink::afterGettingFrame(void* clientData,
                                 unsigned frameSize,
                                 unsigned numTruncatedBytes,
                                 struct timeval presentationTime,
                                 unsigned /*durationInMicroseconds*/) {
    FileSink* sink = (FileSink*)clientData;
    sink->afterGettingFrame(frameSize, numTruncatedBytes, presentationTime);
}

void FileSink::addData(unsigned char const* data,
                       unsigned dataSize,
                       struct timeval presentationTime) {
    if (fPerFrameFileNameBuffer != NULL && fOutFid == NULL) {
        // Special case: Open a new file on-the-fly for this frame
        if (presentationTime.tv_usec == fPrevPresentationTime.tv_usec &&
            presentationTime.tv_sec == fPrevPresentationTime.tv_sec) {
            // The presentation time is unchanged from the previous frame, so we
            // add a 'counter' suffix to the file name, to distinguish them:
            sprintf(fPerFrameFileNameBuffer, "%s-%lu.%06lu-%u",
                    fPerFrameFileNamePrefix, presentationTime.tv_sec,
                    presentationTime.tv_usec, ++fSamePresentationTimeCounter);
        } else {
            sprintf(fPerFrameFileNameBuffer, "%s-%lu.%06lu",
                    fPerFrameFileNamePrefix, presentationTime.tv_sec,
                    presentationTime.tv_usec);
            fPrevPresentationTime = presentationTime;  // for next time
            fSamePresentationTimeCounter = 0;          // for next time
        }
        fOutFid = OpenOutputFile(envir(), fPerFrameFileNameBuffer);
    }

    // Write to our file:
#ifdef TEST_LOSS
    static unsigned const framesPerPacket = 10;
    static unsigned const frameCount = 0;
    static Boolean const packetIsLost;
    if ((frameCount++) % framesPerPacket == 0) {
        packetIsLost =
                (our_random() % 10 == 0);  // simulate 10% packet loss #####
    }

    if (!packetIsLost)
#endif
        if (fOutFid != NULL && data != NULL) {
            fwrite(data, 1, dataSize, fOutFid);
        }
}

void FileSink::afterGettingFrame(unsigned frameSize,
                                 unsigned numTruncatedBytes,
                                 struct timeval presentationTime) {
    if (numTruncatedBytes > 0) {
        envir() << "FileSink::afterGettingFrame(): The input frame data was "
                   "too large for our buffer size ("
                << fBufferSize << ").  " << numTruncatedBytes
                << " bytes of trailing data was dropped!  Correct this by "
                   "increasing the \"bufferSize\" parameter in the "
                   "\"createNew()\" call to at least "
                << fBufferSize + numTruncatedBytes << "\n";
    }
    addData(fBuffer, frameSize, presentationTime);

    if (fOutFid == NULL || fflush(fOutFid) == EOF) {
        // The output file has closed.  Handle this the same way as if the input
        // source had closed:
        if (fSource != NULL) fSource->stopGettingFrames();
        onSourceClosure();
        return;
    }

    if (fPerFrameFileNameBuffer != NULL) {
        if (fOutFid != NULL) {
            fclose(fOutFid);
            fOutFid = NULL;
        }
    }

    // Then try getting the next frame:
    continuePlaying();
}
