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
// 'Ogg' File Sink (recording a single media track only)
// Implementation

#include "OggFileSink.hh"

#include "FramedSource.hh"
#include "MPEG2TransportStreamMultiplexor.hh"  // for calculateCRC()
#include "OutputFile.hh"
#include "VorbisAudioRTPSource.hh"  // for "parseVorbisOrTheoraConfigStr()"

OggFileSink* OggFileSink ::createNew(UsageEnvironment& env,
                                     char const* fileName,
                                     unsigned samplingFrequency,
                                     char const* configStr,
                                     unsigned bufferSize,
                                     Boolean oneFilePerFrame) {
    do {
        FILE* fid;
        char const* perFrameFileNamePrefix;
        if (oneFilePerFrame) {
            // Create the fid for each frame
            fid = NULL;
            perFrameFileNamePrefix = fileName;
        } else {
            // Normal case: create the fid once
            fid = OpenOutputFile(env, fileName);
            if (fid == NULL) break;
            perFrameFileNamePrefix = NULL;
        }

        return new OggFileSink(env, fid, samplingFrequency, configStr,
                               bufferSize, perFrameFileNamePrefix);
    } while (0);

    return NULL;
}

OggFileSink::OggFileSink(UsageEnvironment& env,
                         FILE* fid,
                         unsigned samplingFrequency,
                         char const* configStr,
                         unsigned bufferSize,
                         char const* perFrameFileNamePrefix)
    : FileSink(env, fid, bufferSize, perFrameFileNamePrefix),
      fSamplingFrequency(samplingFrequency),
      fConfigStr(strDup(configStr)),
      fHaveWrittenFirstFrame(False),
      fHaveSeenEOF(False),
      fGranulePosition(0),
      fGranulePositionAdjustment(0),
      fPageSequenceNumber(0),
      fIsTheora(False),
      fGranuleIncrementPerFrame(1),
      fAltFrameSize(0),
      fAltNumTruncatedBytes(0) {
    fAltBuffer = new unsigned char[bufferSize];

    // Initialize our 'Ogg page header' array with constant values:
    u_int8_t* p = fPageHeaderBytes;
    *p++ = 0x4f;
    *p++ = 0x67;
    *p++ = 0x67;
    *p++ = 0x53;  // bytes 0..3: 'capture_pattern': "OggS"
    *p++ = 0;     // byte 4: 'stream_structure_version': 0
    *p++ = 0;     // byte 5: 'header_type_flag': set on each write
    *p++ = 0;
    *p++ = 0;
    *p++ = 0;
    *p++ = 0;
    *p++ = 0;
    *p++ = 0;
    *p++ = 0;
    *p++ = 0;
    // bytes 6..13: 'granule_position': set on each write
    *p++ = 1;
    *p++ = 0;
    *p++ = 0;
    *p++ = 0;  // bytes 14..17: 'bitstream_serial_number': 1
    *p++ = 0;
    *p++ = 0;
    *p++ = 0;
    *p++ = 0;  // bytes 18..21: 'page_sequence_number': set on each write
    *p++ = 0;
    *p++ = 0;
    *p++ = 0;
    *p++ = 0;  // bytes 22..25: 'CRC_checksum': set on each write
    *p = 0;    // byte 26: 'number_page_segments': set on each write
}

OggFileSink::~OggFileSink() {
    // We still have the previously-arrived frame, so write it to the file
    // before we end:
    fHaveSeenEOF = True;
    OggFileSink::addData(fAltBuffer, fAltFrameSize, fAltPresentationTime);

    delete[] fAltBuffer;
    delete[](char*) fConfigStr;
}

Boolean OggFileSink::continuePlaying() {
    // Identical to "FileSink::continuePlaying()",
    // except that we use our own 'on source closure' function:
    if (fSource == NULL) return False;

    fSource->getNextFrame(fBuffer, fBufferSize, FileSink::afterGettingFrame,
                          this, ourOnSourceClosure, this);
    return True;
}

#define PAGE_DATA_MAX_SIZE (255 * 255)

void OggFileSink::addData(unsigned char const* data,
                          unsigned dataSize,
                          struct timeval presentationTime) {
    if (dataSize == 0) return;

    // Set "fGranulePosition" for this frame:
    if (fIsTheora) {
        // Special case for Theora: "fGranulePosition" is supposed to be made up
        // of a pair:
        //   (frame count to last key frame) | (frame count since last key
        //   frame)
        // However, because there appears to be no easy way to figure out which
        // frames are key frames, we just assume that all frames are key frames.
        if (!(data[0] >= 0x80 &&
              data[0] <= 0x82)) {  // for header pages, "fGranulePosition"
                                   // remains 0
            fGranulePosition += fGranuleIncrementPerFrame;
        }
    } else {
        double ptDiff =
                (presentationTime.tv_sec - fFirstPresentationTime.tv_sec) +
                (presentationTime.tv_usec - fFirstPresentationTime.tv_usec) /
                        1000000.0;
        int64_t newGranulePosition = (int64_t)(fSamplingFrequency * ptDiff) +
                                     fGranulePositionAdjustment;
        if (newGranulePosition < fGranulePosition) {
            // Update "fGranulePositionAdjustment" so that "fGranulePosition"
            // remains monotonic
            fGranulePositionAdjustment += fGranulePosition - newGranulePosition;
        } else {
            fGranulePosition = newGranulePosition;
        }
    }

    // Write the frame to the file as a single Ogg 'page' (or perhaps as
    // multiple pages if it's too big for a single page).  We don't aggregate
    // more than one frame within an Ogg page because that's not legal for some
    // headers, and because that would make it difficult for us to properly set
    // the 'eos' (end of stream) flag on the last page.

    // First, figure out how many pages to write here
    // (a page can contain no more than PAGE_DATA_MAX_SIZE bytes)
    unsigned numPagesToWrite = dataSize / PAGE_DATA_MAX_SIZE + 1;
    // Note that if "dataSize" is a integral multiple of PAGE_DATA_MAX_SIZE,
    // there will be an extra 0-size page at the end
    for (unsigned i = 0; i < numPagesToWrite; ++i) {
        // First, fill in the changeable parts of our 'page header' array;
        u_int8_t header_type_flag = 0x0;
        if (!fHaveWrittenFirstFrame && i == 0) {
            header_type_flag |= 0x02;       // 'bos'
            fHaveWrittenFirstFrame = True;  // for the future
        }
        if (i > 0) header_type_flag |= 0x01;  // 'continuation'
        if (fHaveSeenEOF && i == numPagesToWrite - 1)
            header_type_flag |= 0x04;  // 'eos'
        fPageHeaderBytes[5] = header_type_flag;

        if (i < numPagesToWrite - 1) {
            // For pages where the frame does not end, set 'granule_position' in
            // the header to -1:
            fPageHeaderBytes[6] = fPageHeaderBytes[7] = fPageHeaderBytes[8] =
                    fPageHeaderBytes[9] = fPageHeaderBytes[10] =
                            fPageHeaderBytes[11] = fPageHeaderBytes[12] =
                                    fPageHeaderBytes[13] = 0xFF;
        } else {
            fPageHeaderBytes[6] = (u_int8_t)fGranulePosition;
            fPageHeaderBytes[7] = (u_int8_t)(fGranulePosition >> 8);
            fPageHeaderBytes[8] = (u_int8_t)(fGranulePosition >> 16);
            fPageHeaderBytes[9] = (u_int8_t)(fGranulePosition >> 24);
            fPageHeaderBytes[10] = (u_int8_t)(fGranulePosition >> 32);
            fPageHeaderBytes[11] = (u_int8_t)(fGranulePosition >> 40);
            fPageHeaderBytes[12] = (u_int8_t)(fGranulePosition >> 48);
            fPageHeaderBytes[13] = (u_int8_t)(fGranulePosition >> 56);
        }

        fPageHeaderBytes[18] = (u_int8_t)fPageSequenceNumber;
        fPageHeaderBytes[19] = (u_int8_t)(fPageSequenceNumber >> 8);
        fPageHeaderBytes[20] = (u_int8_t)(fPageSequenceNumber >> 16);
        fPageHeaderBytes[21] = (u_int8_t)(fPageSequenceNumber >> 24);
        ++fPageSequenceNumber;

        unsigned pageDataSize;
        u_int8_t number_page_segments;
        if (dataSize >= PAGE_DATA_MAX_SIZE) {
            pageDataSize = PAGE_DATA_MAX_SIZE;
            number_page_segments = 255;
        } else {
            pageDataSize = dataSize;
            number_page_segments =
                    (pageDataSize + 255) /
                    255;  // so that we don't end with a lacing of 255
        }
        fPageHeaderBytes[26] = number_page_segments;

        u_int8_t segment_table[255];
        for (unsigned j = 0; j < (unsigned)(number_page_segments - 1); ++j) {
            segment_table[j] = 255;
        }
        segment_table[number_page_segments - 1] = pageDataSize % 255;

        // Compute the CRC from the 'page header' array, the 'segment_table',
        // and the frame data:
        u_int32_t crc = 0;
        fPageHeaderBytes[22] = fPageHeaderBytes[23] = fPageHeaderBytes[24] =
                fPageHeaderBytes[25] = 0;
        crc = calculateCRC(fPageHeaderBytes, 27, 0);
        crc = calculateCRC(segment_table, number_page_segments, crc);
        crc = calculateCRC(data, pageDataSize, crc);
        fPageHeaderBytes[22] = (u_int8_t)crc;
        fPageHeaderBytes[23] = (u_int8_t)(crc >> 8);
        fPageHeaderBytes[24] = (u_int8_t)(crc >> 16);
        fPageHeaderBytes[25] = (u_int8_t)(crc >> 24);

        // Then write out the 'page header' array:
        FileSink::addData(fPageHeaderBytes, 27, presentationTime);

        // Then write out the 'segment_table':
        FileSink::addData(segment_table, number_page_segments,
                          presentationTime);

        // Then add frame data, to complete the page:
        FileSink::addData(data, pageDataSize, presentationTime);
        data += pageDataSize;
        dataSize -= pageDataSize;
    }
}

void OggFileSink::afterGettingFrame(unsigned frameSize,
                                    unsigned numTruncatedBytes,
                                    struct timeval presentationTime) {
    if (!fHaveWrittenFirstFrame) {
        fFirstPresentationTime = presentationTime;

        // If we have a 'config string' representing 'packed configuration
        // headers'
        // ("identification", "comment", "setup"), unpack them and prepend them
        // to the file:
        if (fConfigStr != NULL && fConfigStr[0] != '\0') {
            u_int8_t* identificationHdr;
            unsigned identificationHdrSize;
            u_int8_t* commentHdr;
            unsigned commentHdrSize;
            u_int8_t* setupHdr;
            unsigned setupHdrSize;
            u_int32_t identField;
            parseVorbisOrTheoraConfigStr(fConfigStr, identificationHdr,
                                         identificationHdrSize, commentHdr,
                                         commentHdrSize, setupHdr, setupHdrSize,
                                         identField);
            if (identificationHdrSize >= 42 &&
                strncmp((const char*)&identificationHdr[1], "theora", 6) == 0) {
                // Hack for Theora video: Parse the "identification" hdr to get
                // the "KFGSHIFT" parameter:
                fIsTheora = True;
                u_int8_t const KFGSHIFT = ((identificationHdr[40] & 3) << 3) |
                                          (identificationHdr[41] >> 5);
                fGranuleIncrementPerFrame = (u_int64_t)(1 << KFGSHIFT);
            }
            OggFileSink::addData(identificationHdr, identificationHdrSize,
                                 presentationTime);
            OggFileSink::addData(commentHdr, commentHdrSize, presentationTime);

            // Hack: Handle the "setup" header as if had arrived in the previous
            // delivery, so it'll get written properly below:
            if (setupHdrSize > fBufferSize) {
                fAltFrameSize = fBufferSize;
                fAltNumTruncatedBytes = setupHdrSize - fBufferSize;
            } else {
                fAltFrameSize = setupHdrSize;
                fAltNumTruncatedBytes = 0;
            }
            memmove(fAltBuffer, setupHdr, fAltFrameSize);
            fAltPresentationTime = presentationTime;

            delete[] identificationHdr;
            delete[] commentHdr;
            delete[] setupHdr;
        }
    }

    // Save this input frame for next time, and instead write the previous input
    // frame now:
    unsigned char* tmpPtr = fBuffer;
    fBuffer = fAltBuffer;
    fAltBuffer = tmpPtr;
    unsigned prevFrameSize = fAltFrameSize;
    fAltFrameSize = frameSize;
    unsigned prevNumTruncatedBytes = fAltNumTruncatedBytes;
    fAltNumTruncatedBytes = numTruncatedBytes;
    struct timeval prevPresentationTime = fAltPresentationTime;
    fAltPresentationTime = presentationTime;

    // Call the parent class to complete the normal file write with the
    // (previous) input frame:
    FileSink::afterGettingFrame(prevFrameSize, prevNumTruncatedBytes,
                                prevPresentationTime);
}

void OggFileSink::ourOnSourceClosure(void* clientData) {
    ((OggFileSink*)clientData)->ourOnSourceClosure();
}

void OggFileSink::ourOnSourceClosure() {
    fHaveSeenEOF = True;

    // We still have the previously-arrived frame, so write it to the file
    // before we end:
    OggFileSink::addData(fAltBuffer, fAltFrameSize, fAltPresentationTime);

    // Handle the closure for real:
    onSourceClosure();
}
