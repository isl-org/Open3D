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
// Demultiplexer for a MPEG Transport Stream
// Implementation

#include "MPEG2TransportStreamDemux.hh"

#include "MPEG2TransportStreamParser.hh"

MPEG2TransportStreamDemux* MPEG2TransportStreamDemux ::createNew(
        UsageEnvironment& env,
        FramedSource* inputSource,
        FramedSource::onCloseFunc* onCloseFunc,
        void* onCloseClientData) {
    return new MPEG2TransportStreamDemux(env, inputSource, onCloseFunc,
                                         onCloseClientData);
}

MPEG2TransportStreamDemux ::MPEG2TransportStreamDemux(
        UsageEnvironment& env,
        FramedSource* inputSource,
        FramedSource::onCloseFunc* onCloseFunc,
        void* onCloseClientData)
    : Medium(env),
      fOnCloseFunc(onCloseFunc),
      fOnCloseClientData(onCloseClientData) {
    fParser =
            new MPEG2TransportStreamParser(inputSource, handleEndOfFile, this);
}

MPEG2TransportStreamDemux::~MPEG2TransportStreamDemux() { delete fParser; }

void MPEG2TransportStreamDemux::handleEndOfFile(void* clientData) {
    ((MPEG2TransportStreamDemux*)clientData)->handleEndOfFile();
}

void MPEG2TransportStreamDemux::handleEndOfFile() {
    if (fOnCloseFunc != NULL) (*fOnCloseFunc)(fOnCloseClientData);
    delete this;
}
