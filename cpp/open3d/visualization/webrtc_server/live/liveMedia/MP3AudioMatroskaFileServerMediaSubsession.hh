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
// (Actually, MPEG-1 or MPEG-2 audio should also work.)
// C++ header

#ifndef _MP3_AUDIO_MATROSKA_FILE_SERVER_MEDIA_SUBSESSION_HH
#define _MP3_AUDIO_MATROSKA_FILE_SERVER_MEDIA_SUBSESSION_HH

#ifndef _MP3_AUDIO_FILE_SERVER_MEDIA_SUBSESSION_HH
#include "MP3AudioFileServerMediaSubsession.hh"
#endif
#ifndef _MATROSKA_FILE_SERVER_DEMUX_HH
#include "MatroskaFileServerDemux.hh"
#endif

class MP3AudioMatroskaFileServerMediaSubsession: public MP3AudioFileServerMediaSubsession {
public:
  static MP3AudioMatroskaFileServerMediaSubsession*
  createNew(MatroskaFileServerDemux& demux, MatroskaTrack* track,
	    Boolean generateADUs = False, Interleaving* interleaving = NULL);
      // Note: "interleaving" is used only if "generateADUs" is True,
      // (and a value of NULL means 'no interleaving')

private:
  MP3AudioMatroskaFileServerMediaSubsession(MatroskaFileServerDemux& demux, MatroskaTrack* track,
					    Boolean generateADUs, Interleaving* interleaving);
      // called only by createNew();
  virtual ~MP3AudioMatroskaFileServerMediaSubsession();

private: // redefined virtual functions
  virtual void seekStreamSource(FramedSource* inputSource, double& seekNPT, double streamDuration, u_int64_t& numBytes);
  virtual FramedSource* createNewStreamSource(unsigned clientSessionId,
                                              unsigned& estBitrate);

private:
  MatroskaFileServerDemux& fOurDemux;
  unsigned fTrackNumber;
};

#endif
