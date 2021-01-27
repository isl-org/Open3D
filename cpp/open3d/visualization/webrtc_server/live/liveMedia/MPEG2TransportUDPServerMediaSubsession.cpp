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
// on demand, from an incoming UDP (or RTP/UDP) MPEG-2 Transport Stream
// Implementation

#include "MPEG2TransportUDPServerMediaSubsession.hh"

#include "BasicUDPSource.hh"
#include "GroupsockHelper.hh"
#include "MPEG2TransportStreamFramer.hh"
#include "SimpleRTPSink.hh"
#include "SimpleRTPSource.hh"

MPEG2TransportUDPServerMediaSubsession*
MPEG2TransportUDPServerMediaSubsession::createNew(UsageEnvironment& env,
                                                  char const* inputAddressStr,
                                                  Port const& inputPort,
                                                  Boolean inputStreamIsRawUDP) {
    return new MPEG2TransportUDPServerMediaSubsession(
            env, inputAddressStr, inputPort, inputStreamIsRawUDP);
}

MPEG2TransportUDPServerMediaSubsession ::MPEG2TransportUDPServerMediaSubsession(
        UsageEnvironment& env,
        char const* inputAddressStr,
        Port const& inputPort,
        Boolean inputStreamIsRawUDP)
    : OnDemandServerMediaSubsession(env, True /*reuseFirstSource*/),
      fInputPort(inputPort),
      fInputGroupsock(NULL),
      fInputStreamIsRawUDP(inputStreamIsRawUDP) {
    fInputAddressStr = strDup(inputAddressStr);
}

MPEG2TransportUDPServerMediaSubsession::
        ~MPEG2TransportUDPServerMediaSubsession() {
    delete fInputGroupsock;
    delete[](char*) fInputAddressStr;
}

FramedSource* MPEG2TransportUDPServerMediaSubsession ::createNewStreamSource(
        unsigned /* clientSessionId*/, unsigned& estBitrate) {
    estBitrate = 5000;  // kbps, estimate

    if (fInputGroupsock == NULL) {
        // Create a 'groupsock' object for receiving the input stream:
        struct in_addr inputAddress;
        inputAddress.s_addr =
                fInputAddressStr == NULL ? 0 : our_inet_addr(fInputAddressStr);
        fInputGroupsock = new Groupsock(envir(), inputAddress, fInputPort, 255);
    }

    FramedSource* transportStreamSource;
    if (fInputStreamIsRawUDP) {
        transportStreamSource =
                BasicUDPSource::createNew(envir(), fInputGroupsock);
    } else {
        transportStreamSource = SimpleRTPSource::createNew(
                envir(), fInputGroupsock, 33, 90000, "video/MP2T", 0,
                False /*no 'M' bit*/);
    }
    return MPEG2TransportStreamFramer::createNew(envir(),
                                                 transportStreamSource);
}

RTPSink* MPEG2TransportUDPServerMediaSubsession ::createNewRTPSink(
        Groupsock* rtpGroupsock,
        unsigned char /*rtpPayloadTypeIfDynamic*/,
        FramedSource* /*inputSource*/) {
    return SimpleRTPSink::createNew(envir(), rtpGroupsock, 33, 90000, "video",
                                    "MP2T", 1, True, False /*no 'M' bit*/);
}
