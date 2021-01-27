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
// A media track, demultiplexed from a MPEG Transport Stream file
// Implementation

#include "MPEG2TransportStreamParser.hh"

MPEG2TransportStreamDemuxedTrack ::MPEG2TransportStreamDemuxedTrack(
        MPEG2TransportStreamParser& ourParser, u_int16_t pid)
    : FramedSource(ourParser.envir()), fOurParser(ourParser), fPID(pid) {}

MPEG2TransportStreamDemuxedTrack::~MPEG2TransportStreamDemuxedTrack() {}

void MPEG2TransportStreamDemuxedTrack::doGetNextFrame() {
    fOurParser.continueParsing();
}
