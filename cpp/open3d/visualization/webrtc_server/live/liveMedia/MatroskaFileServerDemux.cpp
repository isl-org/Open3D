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
// A server demultiplexor for a Matroska file
// Implementation

#include "MatroskaFileServerDemux.hh"

#include "MP3AudioMatroskaFileServerMediaSubsession.hh"
#include "MatroskaFileServerMediaSubsession.hh"

void MatroskaFileServerDemux ::createNew(UsageEnvironment& env,
                                         char const* fileName,
                                         onCreationFunc* onCreation,
                                         void* onCreationClientData,
                                         char const* preferredLanguage) {
    (void)new MatroskaFileServerDemux(env, fileName, onCreation,
                                      onCreationClientData, preferredLanguage);
}

ServerMediaSubsession* MatroskaFileServerDemux::newServerMediaSubsession() {
    unsigned dummyResultTrackNumber;
    return newServerMediaSubsession(dummyResultTrackNumber);
}

ServerMediaSubsession* MatroskaFileServerDemux ::newServerMediaSubsession(
        unsigned& resultTrackNumber) {
    ServerMediaSubsession* result;
    resultTrackNumber = 0;

    for (result = NULL;
         result == NULL && fNextTrackTypeToCheck != MATROSKA_TRACK_TYPE_OTHER;
         fNextTrackTypeToCheck <<= 1) {
        if (fNextTrackTypeToCheck == MATROSKA_TRACK_TYPE_VIDEO)
            resultTrackNumber = fOurMatroskaFile->chosenVideoTrackNumber();
        else if (fNextTrackTypeToCheck == MATROSKA_TRACK_TYPE_AUDIO)
            resultTrackNumber = fOurMatroskaFile->chosenAudioTrackNumber();
        else if (fNextTrackTypeToCheck == MATROSKA_TRACK_TYPE_SUBTITLE)
            resultTrackNumber = fOurMatroskaFile->chosenSubtitleTrackNumber();

        result = newServerMediaSubsessionByTrackNumber(resultTrackNumber);
    }

    return result;
}

ServerMediaSubsession*
MatroskaFileServerDemux ::newServerMediaSubsessionByTrackNumber(
        unsigned trackNumber) {
    MatroskaTrack* track = fOurMatroskaFile->lookup(trackNumber);
    if (track == NULL) return NULL;

    // Use the track's "codecID" string to figure out which
    // "ServerMediaSubsession" subclass to use:
    ServerMediaSubsession* result = NULL;
    if (strcmp(track->mimeType, "audio/MPEG") == 0) {
        result = MP3AudioMatroskaFileServerMediaSubsession::createNew(*this,
                                                                      track);
    } else {
        result = MatroskaFileServerMediaSubsession::createNew(*this, track);
    }

    if (result != NULL) {
#ifdef DEBUG
        fprintf(stderr,
                "Created 'ServerMediaSubsession' object for track #%d: %s "
                "(%s)\n",
                track->trackNumber, track->codecID, track->mimeType);
#endif
    }

    return result;
}

FramedSource* MatroskaFileServerDemux::newDemuxedTrack(unsigned clientSessionId,
                                                       unsigned trackNumber) {
    MatroskaDemux* demuxToUse = NULL;

    if (clientSessionId != 0 && clientSessionId == fLastClientSessionId) {
        demuxToUse =
                fLastCreatedDemux;  // use the same demultiplexor as before
                                    // Note: This code relies upon the fact that
                                    // the creation of streams for different
                                    // client sessions do not overlap - so all
                                    // demuxed tracks are created for one
                                    // "MatroskaDemux" at a time. Also, the
                                    // "clientSessionId != 0" test is a hack,
                                    // because 'session 0' is special; its audio
                                    // and video streams are created and
                                    // destroyed one-at-a-time, rather than both
                                    // streams being created, and then (later)
                                    // both streams being destroyed (as is the
                                    // case for other ('real') session ids).
                                    // Because of this, a separate demultiplexor
                                    // is used for each 'session 0' track.
    }

    if (demuxToUse == NULL) demuxToUse = fOurMatroskaFile->newDemux();

    fLastClientSessionId = clientSessionId;
    fLastCreatedDemux = demuxToUse;

    return demuxToUse->newDemuxedTrackByTrackNumber(trackNumber);
}

MatroskaFileServerDemux ::MatroskaFileServerDemux(UsageEnvironment& env,
                                                  char const* fileName,
                                                  onCreationFunc* onCreation,
                                                  void* onCreationClientData,
                                                  char const* preferredLanguage)
    : Medium(env),
      fFileName(fileName),
      fOnCreation(onCreation),
      fOnCreationClientData(onCreationClientData),
      fNextTrackTypeToCheck(0x1),
      fLastClientSessionId(0),
      fLastCreatedDemux(NULL) {
    MatroskaFile::createNew(env, fileName, onMatroskaFileCreation, this,
                            preferredLanguage);
}

MatroskaFileServerDemux::~MatroskaFileServerDemux() {
    Medium::close(fOurMatroskaFile);
}

void MatroskaFileServerDemux::onMatroskaFileCreation(MatroskaFile* newFile,
                                                     void* clientData) {
    ((MatroskaFileServerDemux*)clientData)->onMatroskaFileCreation(newFile);
}

void MatroskaFileServerDemux::onMatroskaFileCreation(MatroskaFile* newFile) {
    fOurMatroskaFile = newFile;

    // Now, call our own creation notification function:
    if (fOnCreation != NULL) (*fOnCreation)(this, fOnCreationClientData);
}
