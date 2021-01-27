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
// H.265 Video RTP Sources
// C++ header

#ifndef _H265_VIDEO_RTP_SOURCE_HH
#define _H265_VIDEO_RTP_SOURCE_HH

#ifndef _MULTI_FRAMED_RTP_SOURCE_HH
#include "MultiFramedRTPSource.hh"
#endif

class H265VideoRTPSource: public MultiFramedRTPSource {
public:
  static H265VideoRTPSource*
  createNew(UsageEnvironment& env, Groupsock* RTPgs,
	    unsigned char rtpPayloadFormat,
	    Boolean expectDONFields = False,
	    unsigned rtpTimestampFrequency = 90000);
      // "expectDONFields" is True iff we expect incoming H.265/RTP packets to contain
      // DONL and DOND fields.  I.e., if "tx-mode == "MST" or sprop-depack-buf-nalus > 0".

  u_int64_t currentNALUnitAbsDon() const { return fCurrentNALUnitAbsDon; }
      // the 'absolute decoding order number (AbsDon)' for the most-recently delivered NAL unit

protected:
  H265VideoRTPSource(UsageEnvironment& env, Groupsock* RTPgs,
		     unsigned char rtpPayloadFormat,
		     Boolean expectDONFields,		     
		     unsigned rtpTimestampFrequency);
      // called only by createNew()

  virtual ~H265VideoRTPSource();

protected:
  // redefined virtual functions:
  virtual Boolean processSpecialHeader(BufferedPacket* packet,
                                       unsigned& resultSpecialHeaderSize);
  virtual char const* MIMEtype() const;

private:
  void computeAbsDonFromDON(u_int16_t DON);

private:
  friend class H265BufferedPacket;
  Boolean fExpectDONFields;
  unsigned char fCurPacketNALUnitType;
  u_int16_t fPreviousNALUnitDON;
  u_int64_t fCurrentNALUnitAbsDon;
};

#endif
