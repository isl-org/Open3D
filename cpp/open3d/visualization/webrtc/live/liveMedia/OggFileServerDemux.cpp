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
// A server demultiplexor for a Ogg file
// Implementation

#include "OggFileServerDemux.hh"

#include "OggFileServerMediaSubsession.hh"

void OggFileServerDemux ::createNew(UsageEnvironment& env,
                                    char const* fileName,
                                    onCreationFunc* onCreation,
                                    void* onCreationClientData) {
    (void)new OggFileServerDemux(env, fileName, onCreation,
                                 onCreationClientData);
}

ServerMediaSubsession* OggFileServerDemux::newServerMediaSubsession() {
    u_int32_t dummyResultTrackNumber;
    return newServerMediaSubsession(dummyResultTrackNumber);
}

ServerMediaSubsession* OggFileServerDemux ::newServerMediaSubsession(
        u_int32_t& resultTrackNumber) {
    resultTrackNumber = 0;

    OggTrack* nextTrack = fIter->next();
    if (nextTrack == NULL) return NULL;

    return newServerMediaSubsessionByTrackNumber(nextTrack->trackNumber);
}

ServerMediaSubsession*
OggFileServerDemux ::newServerMediaSubsessionByTrackNumber(
        u_int32_t trackNumber) {
    OggTrack* track = fOurOggFile->lookup(trackNumber);
    if (track == NULL) return NULL;

    ServerMediaSubsession* result =
            OggFileServerMediaSubsession::createNew(*this, track);
    if (result != NULL) {
#ifdef DEBUG
        fprintf(stderr,
                "Created 'ServerMediaSubsession' object for track #%d: (%s)\n",
                track->trackNumber, track->mimeType);
#endif
    }

    return result;
}

FramedSource* OggFileServerDemux::newDemuxedTrack(unsigned clientSessionId,
                                                  u_int32_t trackNumber) {
    OggDemux* demuxToUse = NULL;

    if (clientSessionId != 0 && clientSessionId == fLastClientSessionId) {
        demuxToUse =
                fLastCreatedDemux;  // use the same demultiplexor as before
                                    // Note: This code relies upon the fact that
                                    // the creation of streams for different
                                    // client sessions do not overlap - so all
                                    // demuxed tracks are created for one
                                    // "OggDemux" at a time. Also, the
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

    if (demuxToUse == NULL) demuxToUse = fOurOggFile->newDemux();

    fLastClientSessionId = clientSessionId;
    fLastCreatedDemux = demuxToUse;

    return demuxToUse->newDemuxedTrackByTrackNumber(trackNumber);
}

OggFileServerDemux ::OggFileServerDemux(UsageEnvironment& env,
                                        char const* fileName,
                                        onCreationFunc* onCreation,
                                        void* onCreationClientData)
    : Medium(env),
      fFileName(fileName),
      fOnCreation(onCreation),
      fOnCreationClientData(onCreationClientData),
      fIter(NULL /*until the OggFile is created*/),
      fLastClientSessionId(0),
      fLastCreatedDemux(NULL) {
    OggFile::createNew(env, fileName, onOggFileCreation, this);
}

OggFileServerDemux::~OggFileServerDemux() {
    Medium::close(fOurOggFile);

    delete fIter;
}

void OggFileServerDemux::onOggFileCreation(OggFile* newFile, void* clientData) {
    ((OggFileServerDemux*)clientData)->onOggFileCreation(newFile);
}

void OggFileServerDemux::onOggFileCreation(OggFile* newFile) {
    fOurOggFile = newFile;

    fIter = new OggTrackTableIterator(fOurOggFile->trackTable());

    // Now, call our own creation notification function:
    if (fOnCreation != NULL) (*fOnCreation)(this, fOnCreationClientData);
}
