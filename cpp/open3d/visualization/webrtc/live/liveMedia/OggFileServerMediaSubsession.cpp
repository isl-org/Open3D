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
// on demand, from a track within an Ogg file.
// Implementation

#include "OggFileServerMediaSubsession.hh"

#include "FramedFilter.hh"
#include "OggDemuxedTrack.hh"

OggFileServerMediaSubsession* OggFileServerMediaSubsession ::createNew(
        OggFileServerDemux& demux, OggTrack* track) {
    return new OggFileServerMediaSubsession(demux, track);
}

OggFileServerMediaSubsession ::OggFileServerMediaSubsession(
        OggFileServerDemux& demux, OggTrack* track)
    : FileServerMediaSubsession(demux.envir(), demux.fileName(), False),
      fOurDemux(demux),
      fTrack(track),
      fNumFiltersInFrontOfTrack(0) {}

OggFileServerMediaSubsession::~OggFileServerMediaSubsession() {}

FramedSource* OggFileServerMediaSubsession ::createNewStreamSource(
        unsigned clientSessionId, unsigned& estBitrate) {
    FramedSource* baseSource =
            fOurDemux.newDemuxedTrack(clientSessionId, fTrack->trackNumber);
    if (baseSource == NULL) return NULL;

    return fOurDemux.ourOggFile()->createSourceForStreaming(
            baseSource, fTrack->trackNumber, estBitrate,
            fNumFiltersInFrontOfTrack);
}

RTPSink* OggFileServerMediaSubsession ::createNewRTPSink(
        Groupsock* rtpGroupsock,
        unsigned char rtpPayloadTypeIfDynamic,
        FramedSource* /*inputSource*/) {
    return fOurDemux.ourOggFile()->createRTPSinkForTrackNumber(
            fTrack->trackNumber, rtpGroupsock, rtpPayloadTypeIfDynamic);
}
