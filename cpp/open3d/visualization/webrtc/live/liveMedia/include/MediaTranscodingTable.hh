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
// A class that implements a database that can be accessed to create
// "FramedFilter" (subclass) objects that transcode one codec into another.
// The implementation of this class just returns NULL for each codec lookup;
// To actually implement transcoding, you would subclass it.
// C++ header

#ifndef _MEDIA_TRANSCODING_TABLE_HH
#define _MEDIA_TRANSCODING_TABLE_HH

#ifndef _FRAMED_FILTER_HH
#include "FramedFilter.hh"
#endif
#ifndef _MEDIA_SESSION_HH
#include "MediaSession.hh"
#endif

class MediaTranscodingTable: public Medium {
public:
  virtual FramedFilter*
  lookupTranscoder(MediaSubsession& /*inputCodecDescription*/, // in
		   char*& outputCodecName/* out; must be delete[]d later */) {
    // Default implementation: Return NULL (indicating: no transcoding).
    // You would reimplement this virtual function in a subclass to return a new 'transcoding'
    // "FramedFilter" (subclass) object for each ("mediumName","codecName") that you wish to
    // transcode (or return NULL for no transcoding).
    // (Note that "inputCodecDescription" must have a non-NULL "readSource()"; this is used
    //  as the input to the new "FramedFilter" (subclass) object.)
    outputCodecName = NULL;
    return NULL;
  }

  virtual Boolean weWillTranscode(char const* /*mediumName*/, char const* /*codecName*/) {
    // Default implementation: Return False.
    // You would reimplement this in a subclass - returning True for each
    // <mediumName>/<codecName> for which you'll do transcoding.
    // Note: Unlike "lookupTranscoder()", this function does not actually create any 'transcoding'
    // filter objects.  (It may be called before "MediaSubsession::initiate()".)
    return False;
  }

protected: // we are to be subclassed only
  MediaTranscodingTable(UsageEnvironment& env)
    : Medium(env) {
  }
  virtual ~MediaTranscodingTable() {
  }
};

#endif
