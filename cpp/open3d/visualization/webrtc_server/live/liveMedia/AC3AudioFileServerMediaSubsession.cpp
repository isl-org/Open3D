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
// A 'ServerMediaSubsession' object that creates new, unicast, "RTPSink"s
// on demand, from an AC3 audio file.
// Implementation

#include "AC3AudioFileServerMediaSubsession.hh"

#include "AC3AudioRTPSink.hh"
#include "AC3AudioStreamFramer.hh"
#include "ByteStreamFileSource.hh"

AC3AudioFileServerMediaSubsession* AC3AudioFileServerMediaSubsession::createNew(
        UsageEnvironment& env, char const* fileName, Boolean reuseFirstSource) {
    return new AC3AudioFileServerMediaSubsession(env, fileName,
                                                 reuseFirstSource);
}

AC3AudioFileServerMediaSubsession ::AC3AudioFileServerMediaSubsession(
        UsageEnvironment& env, char const* fileName, Boolean reuseFirstSource)
    : FileServerMediaSubsession(env, fileName, reuseFirstSource) {}

AC3AudioFileServerMediaSubsession::~AC3AudioFileServerMediaSubsession() {}

FramedSource* AC3AudioFileServerMediaSubsession ::createNewStreamSource(
        unsigned /*clientSessionId*/, unsigned& estBitrate) {
    estBitrate = 48;  // kbps, estimate

    ByteStreamFileSource* fileSource =
            ByteStreamFileSource::createNew(envir(), fFileName);
    if (fileSource == NULL) return NULL;

    return AC3AudioStreamFramer::createNew(envir(), fileSource);
}

RTPSink* AC3AudioFileServerMediaSubsession ::createNewRTPSink(
        Groupsock* rtpGroupsock,
        unsigned char rtpPayloadTypeIfDynamic,
        FramedSource* inputSource) {
    AC3AudioStreamFramer* audioSource = (AC3AudioStreamFramer*)inputSource;
    return AC3AudioRTPSink::createNew(envir(), rtpGroupsock,
                                      rtpPayloadTypeIfDynamic,
                                      audioSource->samplingRate());
}
