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
// on demand, from an MP3 audio track within a Matroska file.
// (Actually, MPEG-1 or MPEG-2 audio file should also work.)
// Implementation

#include "MP3AudioMatroskaFileServerMediaSubsession.hh"

#include "FileServerMediaSubsession.hh"
#include "MatroskaDemuxedTrack.hh"

MP3AudioMatroskaFileServerMediaSubsession*
MP3AudioMatroskaFileServerMediaSubsession ::createNew(
        MatroskaFileServerDemux& demux,
        MatroskaTrack* track,
        Boolean generateADUs,
        Interleaving* interleaving) {
    return new MP3AudioMatroskaFileServerMediaSubsession(
            demux, track, generateADUs, interleaving);
}

MP3AudioMatroskaFileServerMediaSubsession ::
        MP3AudioMatroskaFileServerMediaSubsession(
                MatroskaFileServerDemux& demux,
                MatroskaTrack* track,
                Boolean generateADUs,
                Interleaving* interleaving)
    : MP3AudioFileServerMediaSubsession(demux.envir(),
                                        demux.fileName(),
                                        False,
                                        generateADUs,
                                        interleaving),
      fOurDemux(demux),
      fTrackNumber(track->trackNumber) {
    fFileDuration = fOurDemux.fileDuration();
}

MP3AudioMatroskaFileServerMediaSubsession::
        ~MP3AudioMatroskaFileServerMediaSubsession() {}

void MP3AudioMatroskaFileServerMediaSubsession ::seekStreamSource(
        FramedSource* inputSource,
        double& seekNPT,
        double /*streamDuration*/,
        u_int64_t& /*numBytes*/) {
    FramedSource* sourceMP3Stream;
    ADUFromMP3Source* aduStream;
    getBaseStreams(inputSource, sourceMP3Stream, aduStream);

    if (aduStream != NULL)
        aduStream->resetInput();  // because we're about to seek within its
                                  // source
    ((MatroskaDemuxedTrack*)sourceMP3Stream)->seekToTime(seekNPT);
}

FramedSource* MP3AudioMatroskaFileServerMediaSubsession ::createNewStreamSource(
        unsigned clientSessionId, unsigned& estBitrate) {
    FramedSource* baseMP3Source =
            fOurDemux.newDemuxedTrack(clientSessionId, fTrackNumber);
    return createNewStreamSourceCommon(baseMP3Source, 0, estBitrate);
}
