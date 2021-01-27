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
// A class that encapsulates an Ogg file
// C++ header

#ifndef _OGG_FILE_HH
#define _OGG_FILE_HH

#ifndef _RTP_SINK_HH
#include "RTPSink.hh"
#endif
#ifndef _HASH_TABLE_HH
#include "HashTable.hh"
#endif

class OggTrack; // forward
class OggDemux; // forward

class OggFile: public Medium {
public:
  typedef void (onCreationFunc)(OggFile* newFile, void* clientData);
  static void createNew(UsageEnvironment& env, char const* fileName,
			onCreationFunc* onCreation, void* onCreationClientData);
      // Note: Unlike most "createNew()" functions, this one doesn't return a new object
      // immediately.  Instead, because this class requires file reading (to parse the
      // Ogg track headers) before a new object can be initialized, the creation of a new object
      // is signalled by calling - from the event loop - an 'onCreationFunc' that is passed as
      // a parameter to "createNew()".

  OggTrack* lookup(u_int32_t trackNumber);

  OggDemux* newDemux();
      // Creates a demultiplexor for extracting tracks from this file.
      // (Separate clients will typically have separate demultiplexors.)

  char const* fileName() const { return fFileName; }
  unsigned numTracks() const;

  FramedSource*
  createSourceForStreaming(FramedSource* baseSource, u_int32_t trackNumber,
                           unsigned& estBitrate, unsigned& numFiltersInFrontOfTrack);
    // Takes a data source (which must be a demultiplexed track from this file) and returns
    // a (possibly modified) data source that can be used for streaming.

  RTPSink* createRTPSinkForTrackNumber(u_int32_t trackNumber, Groupsock* rtpGroupsock,
                                       unsigned char rtpPayloadTypeIfDynamic);
    // Creates a "RTPSink" object that would be appropriate for streaming the specified track,
    // or NULL if no appropriate "RTPSink" exists

  class OggTrackTable& trackTable() { return *fTrackTable; }

private:
  OggFile(UsageEnvironment& env, char const* fileName, onCreationFunc* onCreation, void* onCreationClientData);
    // called only by createNew()
  virtual ~OggFile();

  static void handleEndOfBosPageParsing(void* clientData);
  void handleEndOfBosPageParsing();

  void addTrack(OggTrack* newTrack);
  void removeDemux(OggDemux* demux);

private:
  friend class OggFileParser;
  friend class OggDemux;
  char const* fFileName;
  onCreationFunc* fOnCreation;
  void* fOnCreationClientData;

  class OggTrackTable* fTrackTable;
  HashTable* fDemuxesTable;
  class OggFileParser* fParserForInitialization;
};

class OggTrack {
public:
  OggTrack();
  virtual ~OggTrack();

  // track parameters
  u_int32_t trackNumber; // bitstream serial number
  char const* mimeType; // NULL if not known

  unsigned samplingFrequency, numChannels; // for audio tracks
  unsigned estBitrate; // estimate, in kbps (for RTCP)

  // Special headers for Vorbis audio, Theora video, and Opus audio tracks:
  struct _vtoHdrs {
    u_int8_t* header[3]; // "identification", "comment", "setup"
    unsigned headerSize[3];

    // Fields specific to Vorbis audio:
    unsigned blocksize[2]; // samples per frame (packet)
    unsigned uSecsPerPacket[2]; // computed as (blocksize[i]*1000000)/samplingFrequency
    unsigned vorbis_mode_count;
    unsigned ilog_vorbis_mode_count_minus_1;
    u_int8_t* vorbis_mode_blockflag;
        // an array (of size "vorbis_mode_count") of indexes into the (2-entry) "blocksize" array

    // Fields specific to Theora video:
    u_int8_t KFGSHIFT;
    unsigned uSecsPerFrame;

  } vtoHdrs;

  Boolean weNeedHeaders() const {
    return
      vtoHdrs.header[0] == NULL ||
      vtoHdrs.header[1] == NULL ||
      (vtoHdrs.header[2] == NULL && strcmp(mimeType, "audio/OPUS") != 0);
    }
};

class OggTrackTableIterator {
public:
  OggTrackTableIterator(class OggTrackTable& ourTable);
  virtual ~OggTrackTableIterator();

  OggTrack* next();

private:
  HashTable::Iterator* fIter;
};

class OggDemux: public Medium {
public:
  FramedSource* newDemuxedTrack(u_int32_t& resultTrackNumber);
    // Returns a new stream ("FramedSource" subclass) that represents the next media track
    // from the file.  This function returns NULL when no more media tracks exist.

  FramedSource* newDemuxedTrackByTrackNumber(unsigned trackNumber);
      // As above, but creates a new stream for a specific track number within the Matroska file.
      // (You should not call this function more than once with the same track number.)

      // Note: We assume that:
      // - Every track created by "newDemuxedTrack()" is later read
      // - All calls to "newDemuxedTrack()" are made before any track is read

protected:
  friend class OggFile;
  friend class OggFileParser;
  class OggDemuxedTrack* lookupDemuxedTrack(u_int32_t trackNumber);

  OggDemux(OggFile& ourFile);
  virtual ~OggDemux();

private:
  friend class OggDemuxedTrack;
  void removeTrack(u_int32_t trackNumber);
  void continueReading(); // called by a demuxed track to tell us that it has a pending read ("doGetNextFrame()")

  static void handleEndOfFile(void* clientData);
  void handleEndOfFile();

private:
  OggFile& fOurFile;
  class OggFileParser* fOurParser;
  HashTable* fDemuxedTracksTable;
  OggTrackTableIterator* fIter;
};

#endif
