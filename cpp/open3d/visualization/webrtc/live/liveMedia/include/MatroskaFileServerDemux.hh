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
// C++ header

#ifndef _MATROSKA_FILE_SERVER_DEMUX_HH
#define _MATROSKA_FILE_SERVER_DEMUX_HH

#ifndef _SERVER_MEDIA_SESSION_HH
#include "ServerMediaSession.hh"
#endif

#ifndef _MATROSKA_FILE_HH
#include "MatroskaFile.hh"
#endif

class MatroskaFileServerDemux: public Medium {
public:
  typedef void (onCreationFunc)(MatroskaFileServerDemux* newDemux, void* clientData);
  static void createNew(UsageEnvironment& env, char const* fileName,
			onCreationFunc* onCreation, void* onCreationClientData,
			char const* preferredLanguage = "eng");
    // Note: Unlike most "createNew()" functions, this one doesn't return a new object immediately.  Instead, because this class
    // requires file reading (to parse the Matroska 'Track' headers) before a new object can be initialized, the creation of a new
    // object is signalled by calling - from the event loop - an 'onCreationFunc' that is passed as a parameter to "createNew()". 

  ServerMediaSubsession* newServerMediaSubsession();
  ServerMediaSubsession* newServerMediaSubsession(unsigned& resultTrackNumber);
    // Returns a new "ServerMediaSubsession" object that represents the next preferred media track
    // (video, audio, subtitle - in that order) from the file. (Preferred media tracks are based on the file's language preference.)
    // This function returns NULL when no more media tracks exist.

  ServerMediaSubsession* newServerMediaSubsessionByTrackNumber(unsigned trackNumber);
    // As above, but creates a new "ServerMediaSubsession" object for a specific track number within the Matroska file.
    // (You should not call this function more than once with the same track number.)

  // The following public: member functions are called only by the "ServerMediaSubsession" objects:

  MatroskaFile* ourMatroskaFile() { return fOurMatroskaFile; }
  char const* fileName() const { return fFileName; }
  float fileDuration() const { return fOurMatroskaFile->fileDuration(); }

  FramedSource* newDemuxedTrack(unsigned clientSessionId, unsigned trackNumber);
    // Used by the "ServerMediaSubsession" objects to implement their "createNewStreamSource()" virtual function.

private:
  MatroskaFileServerDemux(UsageEnvironment& env, char const* fileName,
			  onCreationFunc* onCreation, void* onCreationClientData,
			  char const* preferredLanguage);
      // called only by createNew()
  virtual ~MatroskaFileServerDemux();

  static void onMatroskaFileCreation(MatroskaFile* newFile, void* clientData);
  void onMatroskaFileCreation(MatroskaFile* newFile);
private:
  char const* fFileName; 
  onCreationFunc* fOnCreation;
  void* fOnCreationClientData;
  MatroskaFile* fOurMatroskaFile;

  // Used to implement "newServerMediaSubsession()":
  u_int8_t fNextTrackTypeToCheck;

  // Used to set up demuxing, to implement "newDemuxedTrack()":
  unsigned fLastClientSessionId;
  MatroskaDemux* fLastCreatedDemux;
};

#endif
