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
// Vorbis Audio RTP Sources
// C++ header

#ifndef _VORBIS_AUDIO_RTP_SOURCE_HH
#define _VORBIS_AUDIO_RTP_SOURCE_HH

#ifndef _MULTI_FRAMED_RTP_SOURCE_HH
#include "MultiFramedRTPSource.hh"
#endif

class VorbisAudioRTPSource: public MultiFramedRTPSource {
public:
  static VorbisAudioRTPSource*
  createNew(UsageEnvironment& env, Groupsock* RTPgs,
	    unsigned char rtpPayloadFormat,
	    unsigned rtpTimestampFrequency);

  u_int32_t curPacketIdent() const { return fCurPacketIdent; } // The current "Ident" field; only the low-order 24 bits are used

protected:
  VorbisAudioRTPSource(UsageEnvironment& env, Groupsock* RTPgs,
		       unsigned char rtpPayloadFormat,
		       unsigned rtpTimestampFrequency);
      // called only by createNew()

  virtual ~VorbisAudioRTPSource();

protected:
  // redefined virtual functions:
  virtual Boolean processSpecialHeader(BufferedPacket* packet,
                                       unsigned& resultSpecialHeaderSize);
  virtual char const* MIMEtype() const;

private:
  u_int32_t fCurPacketIdent; // only the low-order 24 bits are used
};

void parseVorbisOrTheoraConfigStr(char const* configStr,
				  u_int8_t*& identificationHdr, unsigned& identificationHdrSize,
				  u_int8_t*& commentHdr, unsigned& commentHdrSize,
				  u_int8_t*& setupHdr, unsigned& setupHdrSize,
				  u_int32_t& identField);
    // Returns (in each of the result parameters) unpacked Vorbis or Theora
    // "identification", "comment", and "setup" headers that were specified in a
    // "config" string (in the SDP description for a Vorbis/RTP or Theora/RTP stream).
    // Each of the "*Hdr" result arrays are dynamically allocated by this routine,
    // and must be delete[]d by the caller.

#endif
