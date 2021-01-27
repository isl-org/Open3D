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
// Implementation

#include "H265VideoRTPSource.hh"

////////// H265BufferedPacket and H265BufferedPacketFactory //////////

class H265BufferedPacket : public BufferedPacket {
public:
    H265BufferedPacket(H265VideoRTPSource& ourSource);
    virtual ~H265BufferedPacket();

private:  // redefined virtual functions
    virtual unsigned nextEnclosedFrameSize(unsigned char*& framePtr,
                                           unsigned dataSize);

private:
    H265VideoRTPSource& fOurSource;
};

class H265BufferedPacketFactory : public BufferedPacketFactory {
private:  // redefined virtual functions
    virtual BufferedPacket* createNewPacket(MultiFramedRTPSource* ourSource);
};

///////// H265VideoRTPSource implementation ////////

H265VideoRTPSource* H265VideoRTPSource::createNew(
        UsageEnvironment& env,
        Groupsock* RTPgs,
        unsigned char rtpPayloadFormat,
        Boolean expectDONFields,
        unsigned rtpTimestampFrequency) {
    return new H265VideoRTPSource(env, RTPgs, rtpPayloadFormat, expectDONFields,
                                  rtpTimestampFrequency);
}

H265VideoRTPSource ::H265VideoRTPSource(UsageEnvironment& env,
                                        Groupsock* RTPgs,
                                        unsigned char rtpPayloadFormat,
                                        Boolean expectDONFields,
                                        unsigned rtpTimestampFrequency)
    : MultiFramedRTPSource(env,
                           RTPgs,
                           rtpPayloadFormat,
                           rtpTimestampFrequency,
                           new H265BufferedPacketFactory),
      fExpectDONFields(expectDONFields),
      fPreviousNALUnitDON(0),
      fCurrentNALUnitAbsDon((u_int64_t)(~0)) {}

H265VideoRTPSource::~H265VideoRTPSource() {}

Boolean H265VideoRTPSource ::processSpecialHeader(
        BufferedPacket* packet, unsigned& resultSpecialHeaderSize) {
    unsigned char* headerStart = packet->data();
    unsigned packetSize = packet->dataSize();
    u_int16_t DONL = 0;
    unsigned numBytesToSkip;

    // Check the Payload Header's 'nal_unit_type' for special aggregation or
    // fragmentation packets:
    if (packetSize < 2) return False;
    fCurPacketNALUnitType = (headerStart[0] & 0x7E) >> 1;
    switch (fCurPacketNALUnitType) {
        case 48: {  // Aggregation Packet (AP)
            // We skip over the 2-byte Payload Header, and the DONL header (if
            // any).
            if (fExpectDONFields) {
                if (packetSize < 4) return False;
                DONL = (headerStart[2] << 8) | headerStart[3];
                numBytesToSkip = 4;
            } else {
                numBytesToSkip = 2;
            }
            break;
        }
        case 49: {  // Fragmentation Unit (FU)
            // This NALU begins with the 2-byte Payload Header, the 1-byte FU
            // header, and (optionally) the 2-byte DONL header. If the start bit
            // is set, we reconstruct the original NAL header at the end of
            // these 3 (or 5) bytes, and skip over the first 1 (or 3) bytes.
            if (packetSize < 3) return False;
            u_int8_t startBit = headerStart[2] & 0x80;  // from the FU header
            u_int8_t endBit = headerStart[2] & 0x40;    // from the FU header
            if (startBit) {
                fCurrentPacketBeginsFrame = True;

                u_int8_t nal_unit_type =
                        headerStart[2] &
                        0x3F;  // the last 6 bits of the FU header
                u_int8_t newNALHeader[2];
                newNALHeader[0] =
                        (headerStart[0] & 0x81) | (nal_unit_type << 1);
                newNALHeader[1] = headerStart[1];

                if (fExpectDONFields) {
                    if (packetSize < 5) return False;
                    DONL = (headerStart[3] << 8) | headerStart[4];
                    headerStart[3] = newNALHeader[0];
                    headerStart[4] = newNALHeader[1];
                    numBytesToSkip = 3;
                } else {
                    headerStart[1] = newNALHeader[0];
                    headerStart[2] = newNALHeader[1];
                    numBytesToSkip = 1;
                }
            } else {
                // The start bit is not set, so we skip over all headers:
                fCurrentPacketBeginsFrame = False;
                if (fExpectDONFields) {
                    if (packetSize < 5) return False;
                    DONL = (headerStart[3] << 8) | headerStart[4];
                    numBytesToSkip = 5;
                } else {
                    numBytesToSkip = 3;
                }
            }
            fCurrentPacketCompletesFrame = (endBit != 0);
            break;
        }
        default: {
            // This packet contains one complete NAL unit:
            fCurrentPacketBeginsFrame = fCurrentPacketCompletesFrame = True;
            numBytesToSkip = 0;
            break;
        }
    }

    computeAbsDonFromDON(DONL);
    resultSpecialHeaderSize = numBytesToSkip;
    return True;
}

char const* H265VideoRTPSource::MIMEtype() const { return "video/H265"; }

void H265VideoRTPSource::computeAbsDonFromDON(u_int16_t DON) {
    if (!fExpectDONFields) {
        // Without DON fields in the input stream, we just increment our
        // "AbsDon" count each time:
        ++fCurrentNALUnitAbsDon;
    } else {
        if (fCurrentNALUnitAbsDon == (u_int64_t)(~0)) {
            // This is the very first NAL unit, so "AbsDon" is just "DON":
            fCurrentNALUnitAbsDon = (u_int64_t)DON;
        } else {
            // Use the previous NAL unit's DON and the current DON to compute
            // "AbsDon":
            //     AbsDon[n] = AbsDon[n-1] + (DON[n] - DON[n-1]) mod 2^16
            short signedDiff16 = (short)(DON - fPreviousNALUnitDON);
            int64_t signedDiff64 = (int64_t)signedDiff16;
            fCurrentNALUnitAbsDon += signedDiff64;
        }

        fPreviousNALUnitDON = DON;  // for next time
    }
}

////////// H265BufferedPacket and H265BufferedPacketFactory implementation
/////////////

H265BufferedPacket::H265BufferedPacket(H265VideoRTPSource& ourSource)
    : fOurSource(ourSource) {}

H265BufferedPacket::~H265BufferedPacket() {}

unsigned H265BufferedPacket ::nextEnclosedFrameSize(unsigned char*& framePtr,
                                                    unsigned dataSize) {
    unsigned resultNALUSize = 0;  // if an error occurs

    switch (fOurSource.fCurPacketNALUnitType) {
        case 48: {  // Aggregation Packet (AP)
            if (useCount() > 0) {
                // We're other than the first NAL unit inside this Aggregation
                // Packet. Update our 'decoding order number':
                u_int16_t DONL = 0;
                if (fOurSource.fExpectDONFields) {
                    // There's a 1-byte DOND field next:
                    if (dataSize < 1) break;
                    u_int8_t DOND = framePtr[0];
                    DONL = fOurSource.fPreviousNALUnitDON +
                           (u_int16_t)(DOND + 1);
                    ++framePtr;
                    --dataSize;
                }
                fOurSource.computeAbsDonFromDON(DONL);
            }

            // The next 2 bytes are the NAL unit size:
            if (dataSize < 2) break;
            resultNALUSize = (framePtr[0] << 8) | framePtr[1];
            framePtr += 2;
            break;
        }
        default: {
            // Common case: We use the entire packet data:
            return dataSize;
        }
    }

    return (resultNALUSize <= dataSize) ? resultNALUSize : dataSize;
}

BufferedPacket* H265BufferedPacketFactory ::createNewPacket(
        MultiFramedRTPSource* ourSource) {
    return new H265BufferedPacket((H265VideoRTPSource&)(*ourSource));
}
