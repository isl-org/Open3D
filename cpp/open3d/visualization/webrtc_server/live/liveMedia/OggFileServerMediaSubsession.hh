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
// C++ header

#ifndef _OGG_FILE_SERVER_MEDIA_SUBSESSION_HH
#define _OGG_FILE_SERVER_MEDIA_SUBSESSION_HH

#ifndef _FILE_SERVER_MEDIA_SUBSESSION_HH
#include "FileServerMediaSubsession.hh"
#endif
#ifndef _OGG_FILE_SERVER_DEMUX_HH
#include "OggFileServerDemux.hh"
#endif

class OggFileServerMediaSubsession: public FileServerMediaSubsession {
public:
  static OggFileServerMediaSubsession*
  createNew(OggFileServerDemux& demux, OggTrack* track);

protected:
  OggFileServerMediaSubsession(OggFileServerDemux& demux, OggTrack* track);
      // called only by createNew(), or by subclass constructors
  virtual ~OggFileServerMediaSubsession();

protected: // redefined virtual functions
  virtual FramedSource* createNewStreamSource(unsigned clientSessionId,
					      unsigned& estBitrate);
  virtual RTPSink* createNewRTPSink(Groupsock* rtpGroupsock, unsigned char rtpPayloadTypeIfDynamic, FramedSource* inputSource);

protected:
  OggFileServerDemux& fOurDemux;
  OggTrack* fTrack;
  unsigned fNumFiltersInFrontOfTrack;
};

#endif
