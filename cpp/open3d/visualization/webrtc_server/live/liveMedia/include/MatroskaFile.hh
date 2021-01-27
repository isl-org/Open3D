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
// A class that encapsulates a Matroska file.
// C++ header

#ifndef _MATROSKA_FILE_HH
#define _MATROSKA_FILE_HH

#ifndef _RTP_SINK_HH
#include "RTPSink.hh"
#endif
#ifndef _FILE_SINK_HH
#include "FileSink.hh"
#endif
#ifndef _HASH_TABLE_HH
#include "HashTable.hh"
#endif

class MatroskaTrack; // forward
class MatroskaDemux; // forward

class MatroskaFile: public Medium {
public:
  typedef void (onCreationFunc)(MatroskaFile* newFile, void* clientData);
  static void createNew(UsageEnvironment& env, char const* fileName, onCreationFunc* onCreation, void* onCreationClientData,
			char const* preferredLanguage = "eng");
    // Note: Unlike most "createNew()" functions, this one doesn't return a new object immediately.  Instead, because this class
    // requires file reading (to parse the Matroska 'Track' headers) before a new object can be initialized, the creation of a new
    // object is signalled by calling - from the event loop - an 'onCreationFunc' that is passed as a parameter to "createNew()".

  MatroskaTrack* lookup(unsigned trackNumber) const;

  // Create a demultiplexor for extracting tracks from this file.  (Separate clients will typically have separate demultiplexors.)
  MatroskaDemux* newDemux();

  // Parameters of the file ('Segment'); set when the file is parsed:
  unsigned timecodeScale() { return fTimecodeScale; } // in nanoseconds
  float segmentDuration() { return fSegmentDuration; } // in units of "timecodeScale()"
  float fileDuration(); // in seconds
  
  char const* fileName() const { return fFileName; }

  unsigned chosenVideoTrackNumber() { return fChosenVideoTrackNumber; }
  unsigned chosenAudioTrackNumber() { return fChosenAudioTrackNumber; }
  unsigned chosenSubtitleTrackNumber() { return fChosenSubtitleTrackNumber; }

  FramedSource*
  createSourceForStreaming(FramedSource* baseSource, unsigned trackNumber,
			   unsigned& estBitrate, unsigned& numFiltersInFrontOfTrack);
    // Takes a data source (which must be a demultiplexed track from this file) and returns
    // a (possibly modified) data source that can be used for streaming.

  char const* trackMIMEType(unsigned trackNumber) const;
      // in the form "<medium-name>/<CODEC-NAME>", or NULL if no such track exists

  RTPSink* createRTPSinkForTrackNumber(unsigned trackNumber, Groupsock* rtpGroupsock,
				       unsigned char rtpPayloadTypeIfDynamic);
    // Creates a "RTPSink" object that would be appropriate for streaming the specified track,
    // or NULL if no appropriate "RTPSink" exists

  FileSink* createFileSinkForTrackNumber(unsigned trackNumber, char const* fileName);
    // Creates a "FileSink" object that would be appropriate for recording the contents of
    // the specified track, or NULL if no appropriate "FileSink" exists.

private:
  MatroskaFile(UsageEnvironment& env, char const* fileName, onCreationFunc* onCreation, void* onCreationClientData,
	       char const* preferredLanguage);
      // called only by createNew()
  virtual ~MatroskaFile();

  static void handleEndOfTrackHeaderParsing(void* clientData);
  void handleEndOfTrackHeaderParsing();

  void addTrack(MatroskaTrack* newTrack, unsigned trackNumber);
  void addCuePoint(double cueTime, u_int64_t clusterOffsetInFile, unsigned blockNumWithinCluster);
  Boolean lookupCuePoint(double& cueTime, u_int64_t& resultClusterOffsetInFile, unsigned& resultBlockNumWithinCluster);
  void printCuePoints(FILE* fid);

  void removeDemux(MatroskaDemux* demux);

  void getH264ConfigData(MatroskaTrack const* track,
			 u_int8_t*& sps, unsigned& spsSize,
			 u_int8_t*& pps, unsigned& ppsSize);
    // "sps","pps" are dynamically allocated by this function, and must be delete[]d afterwards
  void getH265ConfigData(MatroskaTrack const* track,
			 u_int8_t*& vps, unsigned& vpsSize,
			 u_int8_t*& sps, unsigned& spsSize,
			 u_int8_t*& pps, unsigned& ppsSize);
    // "vps","sps","pps" are dynamically allocated by this function, and must be delete[]d afterwards

  void getVorbisOrTheoraConfigData(MatroskaTrack const* track,
				   u_int8_t*& identificationHeader, unsigned& identificationHeaderSize,
				   u_int8_t*& commentHeader, unsigned& commentHeaderSize,
				   u_int8_t*& setupHeader, unsigned& setupHeaderSize);
    // "identificationHeader", "commentHeader", "setupHeader" are dynamically allocated by this function, and must be delete[]d afterwards

private:
  friend class MatroskaFileParser;
  friend class MatroskaDemux;
  char const* fFileName;
  onCreationFunc* fOnCreation;
  void* fOnCreationClientData;
  char const* fPreferredLanguage;

  unsigned fTimecodeScale; // in nanoseconds
  float fSegmentDuration; // in units of "fTimecodeScale"
  u_int64_t fSegmentDataOffset, fClusterOffset, fCuesOffset;

  class MatroskaTrackTable* fTrackTable;
  HashTable* fDemuxesTable;
  class CuePoint* fCuePoints;
  unsigned fChosenVideoTrackNumber, fChosenAudioTrackNumber, fChosenSubtitleTrackNumber;
  class MatroskaFileParser* fParserForInitialization;
};

// We define our own track type codes as bits (powers of 2), so we can use the set of track types as a bitmap, representing a set:
// (Note that MATROSKA_TRACK_TYPE_OTHER must be last, and have the largest value.)
#define MATROSKA_TRACK_TYPE_VIDEO 0x01
#define MATROSKA_TRACK_TYPE_AUDIO 0x02
#define MATROSKA_TRACK_TYPE_SUBTITLE 0x04
#define MATROSKA_TRACK_TYPE_OTHER 0x08

class MatroskaTrack {
public:
  MatroskaTrack();
  virtual ~MatroskaTrack();

  // track parameters
  unsigned trackNumber;
  u_int8_t trackType;
  Boolean isEnabled, isDefault, isForced;
  unsigned defaultDuration;
  char* name;
  char* language;
  char* codecID;
  unsigned samplingFrequency;
  unsigned numChannels;
  char const* mimeType;
  unsigned codecPrivateSize;
  u_int8_t* codecPrivate;
  Boolean codecPrivateUsesH264FormatForH265; // a hack specifically for H.265 video tracks
  Boolean codecIsOpus; // a hack for Opus audio
  unsigned headerStrippedBytesSize;
  u_int8_t* headerStrippedBytes;
  char const* colorSampling;
  char const* colorimetry;
  unsigned pixelWidth;
  unsigned pixelHeight;
  unsigned bitDepth;
  unsigned subframeSizeSize; // 0 means: frames do not have subframes (the default behavior)
  Boolean haveSubframes() const { return subframeSizeSize > 0; }
};

class MatroskaDemux: public Medium {
public:
  FramedSource* newDemuxedTrack();
  FramedSource* newDemuxedTrack(unsigned& resultTrackNumber);
      // Returns a new stream ("FramedSource" subclass) that represents the next preferred media
      // track (video, audio, subtitle - in that order) from the file. (Preferred media tracks
      // are based on the file's language preference.)
      // This function returns NULL when no more media tracks exist.

  FramedSource* newDemuxedTrackByTrackNumber(unsigned trackNumber);
      // As above, but creates a new stream for a specific track number within the Matroska file.
      // (You should not call this function more than once with the same track number.)

    // Note: We assume that:
    // - Every track created by "newDemuxedTrack()" is later read
    // - All calls to "newDemuxedTrack()" are made before any track is read

protected:
  friend class MatroskaFile;
  friend class MatroskaFileParser;
  class MatroskaDemuxedTrack* lookupDemuxedTrack(unsigned trackNumber);

  MatroskaDemux(MatroskaFile& ourFile); // we're created only by a "MatroskaFile" (a friend)
  virtual ~MatroskaDemux();

private:
  friend class MatroskaDemuxedTrack;
  void removeTrack(unsigned trackNumber);
  void continueReading(); // called by a demuxed track to tell us that it has a pending read ("doGetNextFrame()")
  void seekToTime(double& seekNPT);

  static void handleEndOfFile(void* clientData);
  void handleEndOfFile();

private:
  MatroskaFile& fOurFile;
  class MatroskaFileParser* fOurParser;
  HashTable* fDemuxedTracksTable;

  // Used to implement "newServerMediaSubsession()":
  u_int8_t fNextTrackTypeToCheck;
};

#endif
