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
// A class that encapsulates an Ogg file.
// Implementation

#include "ByteStreamFileSource.hh"
#include "OggDemuxedTrack.hh"
#include "OggFileParser.hh"
#include "SimpleRTPSink.hh"
#include "TheoraVideoRTPSink.hh"
#include "VorbisAudioRTPSink.hh"

////////// OggTrackTable definition /////////

// For looking up and iterating over the file's tracks:

class OggTrackTable {
public:
    OggTrackTable();
    virtual ~OggTrackTable();

    void add(OggTrack* newTrack);
    OggTrack* lookup(u_int32_t trackNumber);

    unsigned numTracks() const;

private:
    friend class OggTrackTableIterator;
    HashTable* fTable;
};

////////// OggFile implementation //////////

void OggFile::createNew(UsageEnvironment& env,
                        char const* fileName,
                        onCreationFunc* onCreation,
                        void* onCreationClientData) {
    new OggFile(env, fileName, onCreation, onCreationClientData);
}

OggTrack* OggFile::lookup(u_int32_t trackNumber) {
    return fTrackTable->lookup(trackNumber);
}

OggDemux* OggFile::newDemux() {
    OggDemux* demux = new OggDemux(*this);
    fDemuxesTable->Add((char const*)demux, demux);

    return demux;
}

unsigned OggFile::numTracks() const { return fTrackTable->numTracks(); }

FramedSource* OggFile ::createSourceForStreaming(
        FramedSource* baseSource,
        u_int32_t trackNumber,
        unsigned& estBitrate,
        unsigned& numFiltersInFrontOfTrack) {
    if (baseSource == NULL) return NULL;

    FramedSource* result = baseSource;  // by default
    numFiltersInFrontOfTrack = 0;       // by default

    // Look at the track's MIME type to set its estimated bitrate (for use by
    // RTCP). (Later, try to be smarter about figuring out the bitrate.) #####
    // Some MIME types also require adding a special 'framer' in front of the
    // source.
    OggTrack* track = lookup(trackNumber);
    if (track != NULL) {  // should always be true
        estBitrate = track->estBitrate;
    }

    return result;
}

RTPSink* OggFile ::createRTPSinkForTrackNumber(
        u_int32_t trackNumber,
        Groupsock* rtpGroupsock,
        unsigned char rtpPayloadTypeIfDynamic) {
    OggTrack* track = lookup(trackNumber);
    if (track == NULL || track->mimeType == NULL) return NULL;

    RTPSink* result = NULL;  // default value for unknown media types

    if (strcmp(track->mimeType, "audio/VORBIS") == 0) {
        // For Vorbis audio, we use the special "identification", "comment", and
        // "setup" headers that we read when we initially read the headers at
        // the start of the file:
        result = VorbisAudioRTPSink::createNew(
                envir(), rtpGroupsock, rtpPayloadTypeIfDynamic,
                track->samplingFrequency, track->numChannels,
                track->vtoHdrs.header[0], track->vtoHdrs.headerSize[0],
                track->vtoHdrs.header[1], track->vtoHdrs.headerSize[1],
                track->vtoHdrs.header[2], track->vtoHdrs.headerSize[2]);
    } else if (strcmp(track->mimeType, "audio/OPUS") == 0) {
        result = SimpleRTPSink ::createNew(
                envir(), rtpGroupsock, rtpPayloadTypeIfDynamic, 48000, "audio",
                "OPUS", 2, False /*only 1 Opus 'packet' in each RTP packet*/);
    } else if (strcmp(track->mimeType, "video/THEORA") == 0) {
        // For Theora video, we use the special "identification", "comment", and
        // "setup" headers that we read when we initially read the headers at
        // the start of the file:
        result = TheoraVideoRTPSink::createNew(
                envir(), rtpGroupsock, rtpPayloadTypeIfDynamic,
                track->vtoHdrs.header[0], track->vtoHdrs.headerSize[0],
                track->vtoHdrs.header[1], track->vtoHdrs.headerSize[1],
                track->vtoHdrs.header[2], track->vtoHdrs.headerSize[2]);
    }

    return result;
}

OggFile::OggFile(UsageEnvironment& env,
                 char const* fileName,
                 onCreationFunc* onCreation,
                 void* onCreationClientData)
    : Medium(env),
      fFileName(strDup(fileName)),
      fOnCreation(onCreation),
      fOnCreationClientData(onCreationClientData) {
    fTrackTable = new OggTrackTable;
    fDemuxesTable = HashTable::create(ONE_WORD_HASH_KEYS);

    FramedSource* inputSource =
            ByteStreamFileSource::createNew(envir(), fileName);
    if (inputSource == NULL) {
        // The specified input file does not exist!
        fParserForInitialization = NULL;
        handleEndOfBosPageParsing();  // we have no file, and thus no tracks,
                                      // but we still need to signal this
    } else {
        // Initialize ourselves by parsing the file's headers:
        fParserForInitialization = new OggFileParser(
                *this, inputSource, handleEndOfBosPageParsing, this);
    }
}

OggFile::~OggFile() {
    delete fParserForInitialization;

    // Delete any outstanding "OggDemux"s, and the table for them:
    OggDemux* demux;
    while ((demux = (OggDemux*)fDemuxesTable->RemoveNext()) != NULL) {
        delete demux;
    }
    delete fDemuxesTable;
    delete fTrackTable;

    delete[](char*) fFileName;
}

void OggFile::handleEndOfBosPageParsing(void* clientData) {
    ((OggFile*)clientData)->handleEndOfBosPageParsing();
}

void OggFile::handleEndOfBosPageParsing() {
    // Delete our parser, because it's done its job now:
    delete fParserForInitialization;
    fParserForInitialization = NULL;

    // Finally, signal our caller that we've been created and initialized:
    if (fOnCreation != NULL) (*fOnCreation)(this, fOnCreationClientData);
}

void OggFile::addTrack(OggTrack* newTrack) { fTrackTable->add(newTrack); }

void OggFile::removeDemux(OggDemux* demux) {
    fDemuxesTable->Remove((char const*)demux);
}

////////// OggTrackTable implementation /////////

OggTrackTable::OggTrackTable()
    : fTable(HashTable::create(ONE_WORD_HASH_KEYS)) {}

OggTrackTable::~OggTrackTable() {
    // Remove and delete all of our "OggTrack" descriptors, and the hash table
    // itself:
    OggTrack* track;
    while ((track = (OggTrack*)fTable->RemoveNext()) != NULL) {
        delete track;
    }
    delete fTable;
}

void OggTrackTable::add(OggTrack* newTrack) {
    OggTrack* existingTrack = (OggTrack*)fTable->Add(
            reinterpret_cast<char const*>(newTrack->trackNumber), newTrack);
    delete existingTrack;  // if any
}

OggTrack* OggTrackTable::lookup(u_int32_t trackNumber) {
    return (OggTrack*)fTable->Lookup(
            reinterpret_cast<char const*>(trackNumber));
}

unsigned OggTrackTable::numTracks() const { return fTable->numEntries(); }

OggTrackTableIterator::OggTrackTableIterator(OggTrackTable& ourTable) {
    fIter = HashTable::Iterator::create(*(ourTable.fTable));
}

OggTrackTableIterator::~OggTrackTableIterator() { delete fIter; }

OggTrack* OggTrackTableIterator::next() {
    char const* key;
    return (OggTrack*)fIter->next(key);
}

////////// OggTrack implementation //////////

OggTrack::OggTrack()
    : trackNumber(0),
      mimeType(NULL),
      samplingFrequency(48000),
      numChannels(2),
      estBitrate(100) {  // default settings
    vtoHdrs.header[0] = vtoHdrs.header[1] = vtoHdrs.header[2] = NULL;
    vtoHdrs.headerSize[0] = vtoHdrs.headerSize[1] = vtoHdrs.headerSize[2] = 0;

    vtoHdrs.vorbis_mode_count = 0;
    vtoHdrs.vorbis_mode_blockflag = NULL;
}

OggTrack::~OggTrack() {
    delete[] vtoHdrs.header[0];
    delete[] vtoHdrs.header[1];
    delete[] vtoHdrs.header[2];
    delete[] vtoHdrs.vorbis_mode_blockflag;
}

///////// OggDemux implementation /////////

FramedSource* OggDemux::newDemuxedTrack(u_int32_t& resultTrackNumber) {
    OggTrack* nextTrack;
    do {
        nextTrack = fIter->next();
    } while (nextTrack != NULL && nextTrack->mimeType == NULL);

    if (nextTrack == NULL) {  // no more tracks
        resultTrackNumber = 0;
        return NULL;
    }

    resultTrackNumber = nextTrack->trackNumber;
    FramedSource* trackSource =
            new OggDemuxedTrack(envir(), resultTrackNumber, *this);
    fDemuxedTracksTable->Add(reinterpret_cast<char const*>(resultTrackNumber),
                             trackSource);
    return trackSource;
}

FramedSource* OggDemux::newDemuxedTrackByTrackNumber(unsigned trackNumber) {
    if (trackNumber == 0) return NULL;

    FramedSource* trackSource =
            new OggDemuxedTrack(envir(), trackNumber, *this);
    fDemuxedTracksTable->Add(reinterpret_cast<char const*>(trackNumber),
                             trackSource);
    return trackSource;
}

OggDemuxedTrack* OggDemux::lookupDemuxedTrack(u_int32_t trackNumber) {
    return (OggDemuxedTrack*)fDemuxedTracksTable->Lookup(
            reinterpret_cast<char const*>(trackNumber));
}

OggDemux::OggDemux(OggFile& ourFile)
    : Medium(ourFile.envir()),
      fOurFile(ourFile),
      fDemuxedTracksTable(HashTable::create(ONE_WORD_HASH_KEYS)),
      fIter(new OggTrackTableIterator(*fOurFile.fTrackTable)) {
    FramedSource* fileSource =
            ByteStreamFileSource::createNew(envir(), ourFile.fileName());
    fOurParser =
            new OggFileParser(ourFile, fileSource, handleEndOfFile, this, this);
}

OggDemux::~OggDemux() {
    // Begin by acting as if we've reached the end of the source file.
    // This should cause all of our demuxed tracks to get closed.
    handleEndOfFile();

    // Then delete our table of "OggDemuxedTrack"s
    // - but not the "OggDemuxedTrack"s themselves; that should have already
    // happened:
    delete fDemuxedTracksTable;

    delete fIter;
    delete fOurParser;
    fOurFile.removeDemux(this);
}

void OggDemux::removeTrack(u_int32_t trackNumber) {
    fDemuxedTracksTable->Remove(reinterpret_cast<char const*>(trackNumber));
    if (fDemuxedTracksTable->numEntries() == 0) {
        // We no longer have any demuxed tracks, so delete ourselves now:
        delete this;
    }
}

void OggDemux::continueReading() { fOurParser->continueParsing(); }

void OggDemux::handleEndOfFile(void* clientData) {
    ((OggDemux*)clientData)->handleEndOfFile();
}

void OggDemux::handleEndOfFile() {
    // Iterate through all of our 'demuxed tracks', handling 'end of input' on
    // each one. Hack: Because this can cause the hash table to get modified
    // underneath us, we don't call the handlers until after we've first
    // iterated through all of the tracks.
    unsigned numTracks = fDemuxedTracksTable->numEntries();
    if (numTracks == 0) return;
    OggDemuxedTrack** tracks = new OggDemuxedTrack*[numTracks];

    HashTable::Iterator* iter =
            HashTable::Iterator::create(*fDemuxedTracksTable);
    unsigned i;
    char const* trackNumber;

    for (i = 0; i < numTracks; ++i) {
        tracks[i] = (OggDemuxedTrack*)iter->next(trackNumber);
    }
    delete iter;

    for (i = 0; i < numTracks; ++i) {
        if (tracks[i] == NULL) continue;  // sanity check; shouldn't happen
        tracks[i]->handleClosure();
    }

    delete[] tracks;
}
