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
// RTP sink for H.265 video
// Implementation

#include "H265VideoRTPSink.hh"

#include "Base64.hh"
#include "BitVector.hh"
#include "H264VideoRTPSource.hh"  // for "parseSPropParameterSets()"
#include "H265VideoStreamFramer.hh"

////////// H265VideoRTPSink implementation //////////

H265VideoRTPSink ::H265VideoRTPSink(UsageEnvironment& env,
                                    Groupsock* RTPgs,
                                    unsigned char rtpPayloadFormat,
                                    u_int8_t const* vps,
                                    unsigned vpsSize,
                                    u_int8_t const* sps,
                                    unsigned spsSize,
                                    u_int8_t const* pps,
                                    unsigned ppsSize)
    : H264or5VideoRTPSink(265,
                          env,
                          RTPgs,
                          rtpPayloadFormat,
                          vps,
                          vpsSize,
                          sps,
                          spsSize,
                          pps,
                          ppsSize) {}

H265VideoRTPSink::~H265VideoRTPSink() {}

H265VideoRTPSink* H265VideoRTPSink ::createNew(UsageEnvironment& env,
                                               Groupsock* RTPgs,
                                               unsigned char rtpPayloadFormat) {
    return new H265VideoRTPSink(env, RTPgs, rtpPayloadFormat);
}

H265VideoRTPSink* H265VideoRTPSink ::createNew(UsageEnvironment& env,
                                               Groupsock* RTPgs,
                                               unsigned char rtpPayloadFormat,
                                               u_int8_t const* vps,
                                               unsigned vpsSize,
                                               u_int8_t const* sps,
                                               unsigned spsSize,
                                               u_int8_t const* pps,
                                               unsigned ppsSize) {
    return new H265VideoRTPSink(env, RTPgs, rtpPayloadFormat, vps, vpsSize, sps,
                                spsSize, pps, ppsSize);
}

H265VideoRTPSink* H265VideoRTPSink ::createNew(UsageEnvironment& env,
                                               Groupsock* RTPgs,
                                               unsigned char rtpPayloadFormat,
                                               char const* sPropVPSStr,
                                               char const* sPropSPSStr,
                                               char const* sPropPPSStr) {
    u_int8_t* vps = NULL;
    unsigned vpsSize = 0;
    u_int8_t* sps = NULL;
    unsigned spsSize = 0;
    u_int8_t* pps = NULL;
    unsigned ppsSize = 0;

    // Parse each 'sProp' string, extracting and then classifying the NAL
    // unit(s) from each one. We're 'liberal in what we accept'; it's OK if the
    // strings don't contain the NAL unit type implied by their names (or if one
    // or more of the strings encode multiple NAL units).
    SPropRecord* sPropRecords[3];
    unsigned numSPropRecords[3];
    sPropRecords[0] = parseSPropParameterSets(sPropVPSStr, numSPropRecords[0]);
    sPropRecords[1] = parseSPropParameterSets(sPropSPSStr, numSPropRecords[1]);
    sPropRecords[2] = parseSPropParameterSets(sPropPPSStr, numSPropRecords[2]);

    for (unsigned j = 0; j < 3; ++j) {
        SPropRecord* records = sPropRecords[j];
        unsigned numRecords = numSPropRecords[j];

        for (unsigned i = 0; i < numRecords; ++i) {
            if (records[i].sPropLength == 0) continue;  // bad data
            u_int8_t nal_unit_type = ((records[i].sPropBytes[0]) & 0x7E) >> 1;
            if (nal_unit_type == 32 /*VPS*/) {
                vps = records[i].sPropBytes;
                vpsSize = records[i].sPropLength;
            } else if (nal_unit_type == 33 /*SPS*/) {
                sps = records[i].sPropBytes;
                spsSize = records[i].sPropLength;
            } else if (nal_unit_type == 34 /*PPS*/) {
                pps = records[i].sPropBytes;
                ppsSize = records[i].sPropLength;
            }
        }
    }

    H265VideoRTPSink* result =
            new H265VideoRTPSink(env, RTPgs, rtpPayloadFormat, vps, vpsSize,
                                 sps, spsSize, pps, ppsSize);
    delete[] sPropRecords[0];
    delete[] sPropRecords[1];
    delete[] sPropRecords[2];

    return result;
}

Boolean H265VideoRTPSink::sourceIsCompatibleWithUs(MediaSource& source) {
    // Our source must be an appropriate framer:
    return source.isH265VideoStreamFramer();
}

char const* H265VideoRTPSink::auxSDPLine() {
    // Generate a new "a=fmtp:" line each time, using our VPS, SPS and PPS (if
    // we have them), otherwise parameters from our framer source (in case
    // they've changed since the last time that we were called):
    H264or5VideoStreamFramer* framerSource = NULL;
    u_int8_t* vps = fVPS;
    unsigned vpsSize = fVPSSize;
    u_int8_t* sps = fSPS;
    unsigned spsSize = fSPSSize;
    u_int8_t* pps = fPPS;
    unsigned ppsSize = fPPSSize;
    if (vps == NULL || sps == NULL || pps == NULL) {
        // We need to get VPS, SPS and PPS from our framer source:
        if (fOurFragmenter == NULL)
            return NULL;  // we don't yet have a fragmenter (and therefore not a
                          // source)
        framerSource =
                (H264or5VideoStreamFramer*)(fOurFragmenter->inputSource());
        if (framerSource == NULL) return NULL;  // we don't yet have a source

        framerSource->getVPSandSPSandPPS(vps, vpsSize, sps, spsSize, pps,
                                         ppsSize);
        if (vps == NULL || sps == NULL || pps == NULL) {
            return NULL;  // our source isn't ready
        }
    }

    // Set up the "a=fmtp:" SDP line for this stream.
    u_int8_t* vpsWEB =
            new u_int8_t[vpsSize];  // "WEB" means "Without Emulation Bytes"
    unsigned vpsWEBSize =
            removeH264or5EmulationBytes(vpsWEB, vpsSize, vps, vpsSize);
    if (vpsWEBSize < 6 /*'profile_tier_level' offset*/ +
                             12 /*num 'profile_tier_level' bytes*/) {
        // Bad VPS size => assume our source isn't ready
        delete[] vpsWEB;
        return NULL;
    }
    u_int8_t const* profileTierLevelHeaderBytes = &vpsWEB[6];
    unsigned profileSpace =
            profileTierLevelHeaderBytes[0] >> 6;  // general_profile_space
    unsigned profileId =
            profileTierLevelHeaderBytes[0] & 0x1F;  // general_profile_idc
    unsigned tierFlag =
            (profileTierLevelHeaderBytes[0] >> 5) & 0x1;  // general_tier_flag
    unsigned levelId = profileTierLevelHeaderBytes[11];   // general_level_idc
    u_int8_t const* interop_constraints = &profileTierLevelHeaderBytes[5];
    char interopConstraintsStr[100];
    sprintf(interopConstraintsStr, "%02X%02X%02X%02X%02X%02X",
            interop_constraints[0], interop_constraints[1],
            interop_constraints[2], interop_constraints[3],
            interop_constraints[4], interop_constraints[5]);
    delete[] vpsWEB;

    char* sprop_vps = base64Encode((char*)vps, vpsSize);
    char* sprop_sps = base64Encode((char*)sps, spsSize);
    char* sprop_pps = base64Encode((char*)pps, ppsSize);

    char const* fmtpFmt =
            "a=fmtp:%d profile-space=%u"
            ";profile-id=%u"
            ";tier-flag=%u"
            ";level-id=%u"
            ";interop-constraints=%s"
            ";sprop-vps=%s"
            ";sprop-sps=%s"
            ";sprop-pps=%s\r\n";
    unsigned fmtpFmtSize = strlen(fmtpFmt) +
                           3 /* max num chars: rtpPayloadType */ +
                           20   /* max num chars: profile_space */
                           + 20 /* max num chars: profile_id */
                           + 20 /* max num chars: tier_flag */
                           + 20 /* max num chars: level_id */
                           + strlen(interopConstraintsStr) + strlen(sprop_vps) +
                           strlen(sprop_sps) + strlen(sprop_pps);
    char* fmtp = new char[fmtpFmtSize];
    sprintf(fmtp, fmtpFmt, rtpPayloadType(), profileSpace, profileId, tierFlag,
            levelId, interopConstraintsStr, sprop_vps, sprop_sps, sprop_pps);

    delete[] sprop_vps;
    delete[] sprop_sps;
    delete[] sprop_pps;

    delete[] fFmtpSDPLine;
    fFmtpSDPLine = fmtp;
    return fFmtpSDPLine;
}
