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
// Collects a stream of incoming MPEG Transport Stream packets into
//  a chunk sufficiently large to send in a single outgoing (RTP or UDP) packet.
// Implementation

#include "MPEG2TransportStreamAccumulator.hh"

MPEG2TransportStreamAccumulator* MPEG2TransportStreamAccumulator::createNew(
        UsageEnvironment& env,
        FramedSource* inputSource,
        unsigned maxPacketSize) {
    return new MPEG2TransportStreamAccumulator(env, inputSource, maxPacketSize);
}

#ifndef TRANSPORT_PACKET_SIZE
#define TRANSPORT_PACKET_SIZE 188
#endif

MPEG2TransportStreamAccumulator ::MPEG2TransportStreamAccumulator(
        UsageEnvironment& env,
        FramedSource* inputSource,
        unsigned maxPacketSize)
    : FramedFilter(env, inputSource),
      fDesiredPacketSize(maxPacketSize < TRANSPORT_PACKET_SIZE
                                 ? TRANSPORT_PACKET_SIZE
                                 : (maxPacketSize / TRANSPORT_PACKET_SIZE)),
      fNumBytesGathered(0) {}

MPEG2TransportStreamAccumulator::~MPEG2TransportStreamAccumulator() {}

void MPEG2TransportStreamAccumulator::doGetNextFrame() {
    if (fNumBytesGathered >= fDesiredPacketSize) {
        // Complete the delivery to the client:
        fFrameSize = fNumBytesGathered;
        fNumBytesGathered = 0;
        afterGetting(this);
    } else {
        // Ask for more data (delivered directly to the client's buffer);
        fInputSource->getNextFrame(fTo, fMaxSize, afterGettingFrame, this,
                                   FramedSource::handleClosure, this);
    }
}

void MPEG2TransportStreamAccumulator ::afterGettingFrame(
        void* clientData,
        unsigned frameSize,
        unsigned numTruncatedBytes,
        struct timeval presentationTime,
        unsigned durationInMicroseconds) {
    MPEG2TransportStreamAccumulator* accumulator =
            (MPEG2TransportStreamAccumulator*)clientData;
    accumulator->afterGettingFrame1(frameSize, numTruncatedBytes,
                                    presentationTime, durationInMicroseconds);
}

void MPEG2TransportStreamAccumulator ::afterGettingFrame1(
        unsigned frameSize,
        unsigned numTruncatedBytes,
        struct timeval presentationTime,
        unsigned durationInMicroseconds) {
    if (fNumBytesGathered == 0) {  // this is the first frame of the new chunk
        fPresentationTime = presentationTime;
        fDurationInMicroseconds = 0;
    }
    fNumBytesGathered += frameSize;
    fTo += frameSize;
    fMaxSize -= frameSize;
    fDurationInMicroseconds += durationInMicroseconds;

    // Try again to complete delivery:
    doGetNextFrame();
}
