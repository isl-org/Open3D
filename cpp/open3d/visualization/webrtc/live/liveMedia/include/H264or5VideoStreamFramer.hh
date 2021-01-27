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
// A filter that breaks up a H.264 or H.265 Video Elementary Stream into NAL units.
// C++ header

#ifndef _H264_OR_5_VIDEO_STREAM_FRAMER_HH
#define _H264_OR_5_VIDEO_STREAM_FRAMER_HH

#ifndef _MPEG_VIDEO_STREAM_FRAMER_HH
#include "MPEGVideoStreamFramer.hh"
#endif

class H264or5VideoStreamFramer: public MPEGVideoStreamFramer {
public:
  void getVPSandSPSandPPS(u_int8_t*& vps, unsigned& vpsSize,
			  u_int8_t*& sps, unsigned& spsSize,
			  u_int8_t*& pps, unsigned& ppsSize) const {
    // Returns pointers to copies of the most recently seen VPS (video parameter set)
    // SPS (sequence parameter set) and PPS (picture parameter set) NAL units.
    // (NULL pointers are returned if the NAL units have not yet been seen.)
    vps = fLastSeenVPS; vpsSize = fLastSeenVPSSize;
    sps = fLastSeenSPS; spsSize = fLastSeenSPSSize;
    pps = fLastSeenPPS; ppsSize = fLastSeenPPSSize;
  }

  void setVPSandSPSandPPS(u_int8_t* vps, unsigned vpsSize,
			  u_int8_t* sps, unsigned spsSize,
			  u_int8_t* pps, unsigned ppsSize) {
    // Assigns copies of the VPS, SPS and PPS NAL units.  If this function is not called,
    // then these NAL units are assigned only if/when they appear in the input stream.
    saveCopyOfVPS(vps, vpsSize);
    saveCopyOfSPS(sps, spsSize);
    saveCopyOfPPS(pps, ppsSize);
  }

protected:
  H264or5VideoStreamFramer(int hNumber, // 264 or 265
			   UsageEnvironment& env, FramedSource* inputSource,
			   Boolean createParser,
			   Boolean includeStartCodeInOutput, Boolean insertAccessUnitDelimiters);
      // We're an abstract base class.
  virtual ~H264or5VideoStreamFramer();

  void saveCopyOfVPS(u_int8_t* from, unsigned size);
  void saveCopyOfSPS(u_int8_t* from, unsigned size);
  void saveCopyOfPPS(u_int8_t* from, unsigned size);

  void setPresentationTime() { fPresentationTime = fNextPresentationTime; }

  Boolean isVPS(u_int8_t nal_unit_type);
  Boolean isSPS(u_int8_t nal_unit_type);
  Boolean isPPS(u_int8_t nal_unit_type);
  Boolean isVCL(u_int8_t nal_unit_type);

protected: // redefined virtual functions
  virtual void doGetNextFrame();

protected:
  int fHNumber;
  Boolean fIncludeStartCodeInOutput, fInsertAccessUnitDelimiters;
  u_int8_t* fLastSeenVPS;
  unsigned fLastSeenVPSSize;
  u_int8_t* fLastSeenSPS;
  unsigned fLastSeenSPSSize;
  u_int8_t* fLastSeenPPS;
  unsigned fLastSeenPPSSize;
  struct timeval fNextPresentationTime; // the presentation time to be used for the next NAL unit to be parsed/delivered after this
  friend class H264or5VideoStreamParser; // hack
};

// A general routine for making a copy of a (H.264 or H.265) NAL unit,
// removing 'emulation' bytes from the copy:
unsigned removeH264or5EmulationBytes(u_int8_t* to, unsigned toMaxSize,
				     u_int8_t const* from, unsigned fromSize);
    // returns the size of the copy; it will be <= min(toMaxSize,fromSize)

#endif
