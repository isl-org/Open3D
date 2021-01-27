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
// on demand, from a track within a Matroska file.
// C++ header

#ifndef _MATROSKA_FILE_SERVER_MEDIA_SUBSESSION_HH
#define _MATROSKA_FILE_SERVER_MEDIA_SUBSESSION_HH

#ifndef _FILE_SERVER_MEDIA_SUBSESSION_HH
#include "FileServerMediaSubsession.hh"
#endif
#ifndef _MATROSKA_FILE_SERVER_DEMUX_HH
#include "MatroskaFileServerDemux.hh"
#endif

class MatroskaFileServerMediaSubsession: public FileServerMediaSubsession {
public:
  static MatroskaFileServerMediaSubsession*
  createNew(MatroskaFileServerDemux& demux, MatroskaTrack* track);

protected:
  MatroskaFileServerMediaSubsession(MatroskaFileServerDemux& demux, MatroskaTrack* track);
      // called only by createNew(), or by subclass constructors
  virtual ~MatroskaFileServerMediaSubsession();

protected: // redefined virtual functions
  virtual float duration() const;
  virtual void seekStreamSource(FramedSource* inputSource, double& seekNPT, double streamDuration, u_int64_t& numBytes);
  virtual FramedSource* createNewStreamSource(unsigned clientSessionId,
					      unsigned& estBitrate);
  virtual RTPSink* createNewRTPSink(Groupsock* rtpGroupsock, unsigned char rtpPayloadTypeIfDynamic, FramedSource* inputSource);

protected:
  MatroskaFileServerDemux& fOurDemux;
  MatroskaTrack* fTrack;
  unsigned fNumFiltersInFrontOfTrack;
};

#endif
