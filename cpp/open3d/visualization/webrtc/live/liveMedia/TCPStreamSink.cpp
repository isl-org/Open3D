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
// A sink representing a TCP output stream
// Implementation

#include "TCPStreamSink.hh"

#include <GroupsockHelper.hh>  // for "ignoreSigPipeOnSocket()"

TCPStreamSink* TCPStreamSink::createNew(UsageEnvironment& env, int socketNum) {
    return new TCPStreamSink(env, socketNum);
}

TCPStreamSink::TCPStreamSink(UsageEnvironment& env, int socketNum)
    : MediaSink(env),
      fUnwrittenBytesStart(0),
      fUnwrittenBytesEnd(0),
      fInputSourceIsOpen(False),
      fOutputSocketIsWritable(True),
      fOutputSocketNum(socketNum) {
    ignoreSigPipeOnSocket(socketNum);
}

TCPStreamSink::~TCPStreamSink() {
    // Turn off any pending background handling of our output socket:
    envir().taskScheduler().disableBackgroundHandling(fOutputSocketNum);
}

Boolean TCPStreamSink::continuePlaying() {
    fInputSourceIsOpen = fSource != NULL;
    processBuffer();

    return True;
}

#define TCP_STREAM_SINK_MIN_READ_SIZE 1000

void TCPStreamSink::processBuffer() {
    // First, try writing data to our output socket, if we can:
    if (fOutputSocketIsWritable && numUnwrittenBytes() > 0) {
        int numBytesWritten = send(fOutputSocketNum,
                                   (const char*)&fBuffer[fUnwrittenBytesStart],
                                   numUnwrittenBytes(), 0);
        if (numBytesWritten < (int)numUnwrittenBytes()) {
            // The output socket is no longer writable.  Set a handler to be
            // called when it becomes writable again.
            fOutputSocketIsWritable = False;
            if (envir().getErrno() !=
                EPIPE) {  // on this error, the socket might still be writable,
                          // but no longer usable
                envir().taskScheduler().setBackgroundHandling(
                        fOutputSocketNum, SOCKET_WRITABLE,
                        socketWritableHandler, this);
            }
        }
        if (numBytesWritten > 0) {
            // We wrote at least some of our data.  Update our buffer pointers:
            fUnwrittenBytesStart += numBytesWritten;
            if (fUnwrittenBytesStart > fUnwrittenBytesEnd)
                fUnwrittenBytesStart = fUnwrittenBytesEnd;  // sanity check
            if (fUnwrittenBytesStart == fUnwrittenBytesEnd &&
                (!fInputSourceIsOpen || !fSource->isCurrentlyAwaitingData())) {
                fUnwrittenBytesStart = fUnwrittenBytesEnd =
                        0;  // reset the buffer to empty
            }
        }
    }

    // Then, read from our input source, if we can (& we're not already reading
    // from it):
    if (fInputSourceIsOpen &&
        freeBufferSpace() >= TCP_STREAM_SINK_MIN_READ_SIZE &&
        !fSource->isCurrentlyAwaitingData()) {
        fSource->getNextFrame(&fBuffer[fUnwrittenBytesEnd], freeBufferSpace(),
                              afterGettingFrame, this, ourOnSourceClosure,
                              this);
    } else if (!fInputSourceIsOpen && numUnwrittenBytes() == 0) {
        // We're now done:
        onSourceClosure();
    }
}

void TCPStreamSink::socketWritableHandler(void* clientData, int /*mask*/) {
    TCPStreamSink* sink = (TCPStreamSink*)clientData;
    sink->socketWritableHandler1();
}

void TCPStreamSink::socketWritableHandler1() {
    envir().taskScheduler().disableBackgroundHandling(
            fOutputSocketNum);  // disable this handler until the next time it's
                                // needed

    fOutputSocketIsWritable = True;
    processBuffer();
}

void TCPStreamSink::afterGettingFrame(void* clientData,
                                      unsigned frameSize,
                                      unsigned numTruncatedBytes,
                                      struct timeval /*presentationTime*/,
                                      unsigned /*durationInMicroseconds*/) {
    TCPStreamSink* sink = (TCPStreamSink*)clientData;
    sink->afterGettingFrame(frameSize, numTruncatedBytes);
}

void TCPStreamSink::afterGettingFrame(unsigned frameSize,
                                      unsigned numTruncatedBytes) {
    if (numTruncatedBytes > 0) {
        envir() << "TCPStreamSink::afterGettingFrame(): The input frame data "
                   "was too large for our buffer.  "
                << numTruncatedBytes
                << " bytes of trailing data was dropped!  Correct this by "
                   "increasing the definition of "
                   "\"TCP_STREAM_SINK_BUFFER_SIZE\" in "
                   "\"include/TCPStreamSink.hh\".\n";
    }
    fUnwrittenBytesEnd += frameSize;
    processBuffer();
}

void TCPStreamSink::ourOnSourceClosure(void* clientData) {
    TCPStreamSink* sink = (TCPStreamSink*)clientData;
    sink->ourOnSourceClosure1();
}

void TCPStreamSink::ourOnSourceClosure1() {
    // The input source has closed:
    fInputSourceIsOpen = False;
    processBuffer();
}
