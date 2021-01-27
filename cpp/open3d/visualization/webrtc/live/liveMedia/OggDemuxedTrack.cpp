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
// A media track, demultiplexed from an Ogg file
// Implementation

#include "OggDemuxedTrack.hh"

#include "OggFile.hh"

OggDemuxedTrack::OggDemuxedTrack(UsageEnvironment& env,
                                 unsigned trackNumber,
                                 OggDemux& sourceDemux)
    : FramedSource(env),
      fOurTrackNumber(trackNumber),
      fOurSourceDemux(sourceDemux),
      fCurrentPageIsContinuation(False) {
    fNextPresentationTime.tv_sec = 0;
    fNextPresentationTime.tv_usec = 0;
}

OggDemuxedTrack::~OggDemuxedTrack() {
    fOurSourceDemux.removeTrack(fOurTrackNumber);
}

void OggDemuxedTrack::doGetNextFrame() { fOurSourceDemux.continueReading(); }

char const* OggDemuxedTrack::MIMEtype() const {
    OggTrack* track = fOurSourceDemux.fOurFile.lookup(fOurTrackNumber);
    if (track == NULL) return "(unknown)";  // shouldn't happen
    return track->mimeType;
}
