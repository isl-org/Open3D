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
// A class that encapsulates a Matroska file.
// Implementation

#include <AC3AudioRTPSink.hh>
#include <AMRAudioFileSink.hh>
#include <Base64.hh>
#include <ByteStreamFileSource.hh>
#include <H264VideoFileSink.hh>
#include <H264VideoRTPSink.hh>
#include <H264VideoStreamDiscreteFramer.hh>
#include <H265VideoFileSink.hh>
#include <H265VideoRTPSink.hh>
#include <H265VideoStreamDiscreteFramer.hh>
#include <MPEG1or2AudioRTPSink.hh>
#include <MPEG4GenericRTPSink.hh>
#include <OggFileSink.hh>
#include <RawVideoRTPSink.hh>
#include <SimpleRTPSink.hh>
#include <T140TextRTPSink.hh>
#include <TheoraVideoRTPSink.hh>
#include <VP8VideoRTPSink.hh>
#include <VP9VideoRTPSink.hh>
#include <VorbisAudioRTPSink.hh>
#include <string>

#include "MatroskaDemuxedTrack.hh"
#include "MatroskaFileParser.hh"

////////// CuePoint definition //////////

class CuePoint {
public:
    CuePoint(double cueTime,
             u_int64_t clusterOffsetInFile,
             unsigned blockNumWithinCluster /* 1-based */);
    virtual ~CuePoint();

    static void addCuePoint(CuePoint*& root,
                            double cueTime,
                            u_int64_t clusterOffsetInFile,
                            unsigned blockNumWithinCluster /* 1-based */,
                            Boolean& needToReviseBalanceOfParent);
    // If "cueTime" == "root.fCueTime", replace the existing data, otherwise add
    // to the left or right subtree. (Note that this is a static member function
    // because - as a result of tree rotation - "root" might change.)

    Boolean lookup(double& cueTime,
                   u_int64_t& resultClusterOffsetInFile,
                   unsigned& resultBlockNumWithinCluster);

    static void fprintf(FILE* fid,
                        CuePoint* cuePoint);  // used for debugging; it's static
                                              // to allow for "cuePoint == NULL"

private:
    // The "CuePoint" tree is implemented as an AVL Tree, to keep it balanced
    // (for efficient lookup).
    CuePoint* fSubTree[2];  // 0 => left; 1 => right
    CuePoint* left() const { return fSubTree[0]; }
    CuePoint* right() const { return fSubTree[1]; }
    char fBalance;  // height of right subtree - height of left subtree

    static void rotate(unsigned direction /*0 => left; 1 => right*/,
                       CuePoint*& root);  // used to keep the tree in balance

    double fCueTime;
    u_int64_t fClusterOffsetInFile;
    unsigned fBlockNumWithinCluster;  // 0-based
};

UsageEnvironment& operator<<(UsageEnvironment& env,
                             const CuePoint* cuePoint);  // used for debugging

////////// MatroskaTrackTable definition /////////

// For looking up and iterating over the file's tracks:
class MatroskaTrackTable {
public:
    MatroskaTrackTable();
    virtual ~MatroskaTrackTable();

    void add(MatroskaTrack* newTrack, unsigned trackNumber);
    MatroskaTrack* lookup(unsigned trackNumber);

    unsigned numTracks() const;

    class Iterator {
    public:
        Iterator(MatroskaTrackTable& ourTable);
        virtual ~Iterator();
        MatroskaTrack* next();

    private:
        HashTable::Iterator* fIter;
    };

private:
    friend class Iterator;
    HashTable* fTable;
};

////////// MatroskaFile implementation //////////

void MatroskaFile ::createNew(UsageEnvironment& env,
                              char const* fileName,
                              onCreationFunc* onCreation,
                              void* onCreationClientData,
                              char const* preferredLanguage) {
    new MatroskaFile(env, fileName, onCreation, onCreationClientData,
                     preferredLanguage);
}

MatroskaFile::MatroskaFile(UsageEnvironment& env,
                           char const* fileName,
                           onCreationFunc* onCreation,
                           void* onCreationClientData,
                           char const* preferredLanguage)
    : Medium(env),
      fFileName(strDup(fileName)),
      fOnCreation(onCreation),
      fOnCreationClientData(onCreationClientData),
      fPreferredLanguage(strDup(preferredLanguage)),
      fTimecodeScale(1000000),
      fSegmentDuration(0.0),
      fSegmentDataOffset(0),
      fClusterOffset(0),
      fCuesOffset(0),
      fCuePoints(NULL),
      fChosenVideoTrackNumber(0),
      fChosenAudioTrackNumber(0),
      fChosenSubtitleTrackNumber(0) {
    fTrackTable = new MatroskaTrackTable;
    fDemuxesTable = HashTable::create(ONE_WORD_HASH_KEYS);

    FramedSource* inputSource =
            ByteStreamFileSource::createNew(envir(), fileName);
    if (inputSource == NULL) {
        // The specified input file does not exist!
        fParserForInitialization = NULL;
        handleEndOfTrackHeaderParsing();  // we have no file, and thus no
                                          // tracks, but we still need to signal
                                          // this
    } else {
        // Initialize ourselves by parsing the file's 'Track' headers:
        fParserForInitialization = new MatroskaFileParser(
                *this, inputSource, handleEndOfTrackHeaderParsing, this, NULL);
    }
}

MatroskaFile::~MatroskaFile() {
    delete fParserForInitialization;
    delete fCuePoints;

    // Delete any outstanding "MatroskaDemux"s, and the table for them:
    MatroskaDemux* demux;
    while ((demux = (MatroskaDemux*)fDemuxesTable->RemoveNext()) != NULL) {
        delete demux;
    }
    delete fDemuxesTable;
    delete fTrackTable;

    delete[](char*) fPreferredLanguage;
    delete[](char*) fFileName;
}

void MatroskaFile::handleEndOfTrackHeaderParsing(void* clientData) {
    ((MatroskaFile*)clientData)->handleEndOfTrackHeaderParsing();
}

class TrackChoiceRecord {
public:
    unsigned trackNumber;
    u_int8_t trackType;
    unsigned choiceFlags;
};

void MatroskaFile::handleEndOfTrackHeaderParsing() {
    // Having parsed all of our track headers, iterate through the tracks to
    // figure out which ones should be played. The Matroska 'specification' is
    // rather imprecise about this (as usual).  However, we use the following
    // algorithm:
    // - Use one (but no more) enabled track of each type (video, audio,
    // subtitle).  (Ignore all tracks that are not 'enabled'.)
    // - For each track type, choose the one that's 'forced'.
    //     - If more than one is 'forced', choose the first one that matches our
    //     preferred language, or the first if none matches.
    //     - If none is 'forced', choose the one that's 'default'.
    //     - If more than one is 'default', choose the first one that matches
    //     our preferred language, or the first if none matches.
    //     - If none is 'default', choose the first one that matches our
    //     preferred language, or the first if none matches.
    unsigned numTracks = fTrackTable->numTracks();
    if (numTracks > 0) {
        TrackChoiceRecord* trackChoice = new TrackChoiceRecord[numTracks];
        unsigned numEnabledTracks = 0;
        MatroskaTrackTable::Iterator iter(*fTrackTable);
        MatroskaTrack* track;
        while ((track = iter.next()) != NULL) {
            if (!track->isEnabled || track->trackType == 0 ||
                track->mimeType[0] == '\0')
                continue;  // track not enabled, or not fully-defined

            trackChoice[numEnabledTracks].trackNumber = track->trackNumber;
            trackChoice[numEnabledTracks].trackType = track->trackType;

            // Assign flags for this track so that, when sorted, the largest
            // value becomes our choice:
            unsigned choiceFlags = 0;
            if (fPreferredLanguage != NULL && track->language != NULL &&
                strcmp(fPreferredLanguage, track->language) == 0) {
                // This track matches our preferred language:
                choiceFlags |= 1;
            }
            if (track->isForced) {
                choiceFlags |= 4;
            } else if (track->isDefault) {
                choiceFlags |= 2;
            }
            trackChoice[numEnabledTracks].choiceFlags = choiceFlags;

            ++numEnabledTracks;
        }

        // Choose the desired track for each track type:
        for (u_int8_t trackType = 0x01; trackType != MATROSKA_TRACK_TYPE_OTHER;
             trackType <<= 1) {
            int bestNum = -1;
            int bestChoiceFlags = -1;
            for (unsigned i = 0; i < numEnabledTracks; ++i) {
                if (trackChoice[i].trackType == trackType &&
                    (int)trackChoice[i].choiceFlags > bestChoiceFlags) {
                    bestNum = i;
                    bestChoiceFlags = (int)trackChoice[i].choiceFlags;
                }
            }
            if (bestChoiceFlags >= 0) {  // There is a track for this track type
                if (trackType == MATROSKA_TRACK_TYPE_VIDEO)
                    fChosenVideoTrackNumber = trackChoice[bestNum].trackNumber;
                else if (trackType == MATROSKA_TRACK_TYPE_AUDIO)
                    fChosenAudioTrackNumber = trackChoice[bestNum].trackNumber;
                else
                    fChosenSubtitleTrackNumber =
                            trackChoice[bestNum].trackNumber;
            }
        }

        delete[] trackChoice;
    }

#ifdef DEBUG
    if (fChosenVideoTrackNumber > 0)
        fprintf(stderr, "Chosen video track: #%d\n", fChosenVideoTrackNumber);
    else
        fprintf(stderr, "No chosen video track\n");
    if (fChosenAudioTrackNumber > 0)
        fprintf(stderr, "Chosen audio track: #%d\n", fChosenAudioTrackNumber);
    else
        fprintf(stderr, "No chosen audio track\n");
    if (fChosenSubtitleTrackNumber > 0)
        fprintf(stderr, "Chosen subtitle track: #%d\n",
                fChosenSubtitleTrackNumber);
    else
        fprintf(stderr, "No chosen subtitle track\n");
#endif

    // Delete our parser, because it's done its job now:
    delete fParserForInitialization;
    fParserForInitialization = NULL;

    // Finally, signal our caller that we've been created and initialized:
    if (fOnCreation != NULL) (*fOnCreation)(this, fOnCreationClientData);
}

MatroskaTrack* MatroskaFile::lookup(unsigned trackNumber) const {
    return fTrackTable->lookup(trackNumber);
}

MatroskaDemux* MatroskaFile::newDemux() {
    MatroskaDemux* demux = new MatroskaDemux(*this);
    fDemuxesTable->Add((char const*)demux, demux);

    return demux;
}

void MatroskaFile::removeDemux(MatroskaDemux* demux) {
    fDemuxesTable->Remove((char const*)demux);
}

#define getPrivByte(b) \
    if (n == 0)        \
        break;         \
    else               \
        do {           \
            b = *p++;  \
            --n;       \
        } while (0) /* Vorbis/Theora configuration header parsing */
#define CHECK_PTR \
    if (ptr >= limit) break                         /* H.264/H.265 parsing */
#define NUM_BYTES_REMAINING (unsigned)(limit - ptr) /* H.264/H.265 parsing */

void MatroskaFile::getH264ConfigData(MatroskaTrack const* track,
                                     u_int8_t*& sps,
                                     unsigned& spsSize,
                                     u_int8_t*& pps,
                                     unsigned& ppsSize) {
    sps = pps = NULL;
    spsSize = ppsSize = 0;

    do {
        if (track == NULL) break;

        // Use our track's 'Codec Private' data: Bytes 5 and beyond contain SPS
        // and PPSs:
        if (track->codecPrivateSize < 6) break;
        u_int8_t* SPSandPPSBytes = &track->codecPrivate[5];
        unsigned numSPSandPPSBytes = track->codecPrivateSize - 5;

        // Extract, from "SPSandPPSBytes", one SPS NAL unit, and one PPS NAL
        // unit. (I hope one is all we need of each.)
        unsigned i;
        u_int8_t* ptr = SPSandPPSBytes;
        u_int8_t* limit = &SPSandPPSBytes[numSPSandPPSBytes];

        unsigned numSPSs = (*ptr++) & 0x1F;
        CHECK_PTR;
        for (i = 0; i < numSPSs; ++i) {
            unsigned spsSize1 = (*ptr++) << 8;
            CHECK_PTR;
            spsSize1 |= *ptr++;
            CHECK_PTR;

            if (spsSize1 > NUM_BYTES_REMAINING) break;
            u_int8_t nal_unit_type = ptr[0] & 0x1F;
            if (sps == NULL &&
                nal_unit_type == 7 /*sanity check*/) {  // save the first one
                spsSize = spsSize1;
                sps = new u_int8_t[spsSize];
                memmove(sps, ptr, spsSize);
            }
            ptr += spsSize1;
        }

        unsigned numPPSs = (*ptr++) & 0x1F;
        CHECK_PTR;
        for (i = 0; i < numPPSs; ++i) {
            unsigned ppsSize1 = (*ptr++) << 8;
            CHECK_PTR;
            ppsSize1 |= *ptr++;
            CHECK_PTR;

            if (ppsSize1 > NUM_BYTES_REMAINING) break;
            u_int8_t nal_unit_type = ptr[0] & 0x1F;
            if (pps == NULL &&
                nal_unit_type == 8 /*sanity check*/) {  // save the first one
                ppsSize = ppsSize1;
                pps = new u_int8_t[ppsSize];
                memmove(pps, ptr, ppsSize);
            }
            ptr += ppsSize1;
        }

        return;
    } while (0);

    // An error occurred:
    delete[] sps;
    sps = NULL;
    spsSize = 0;
    delete[] pps;
    pps = NULL;
    ppsSize = 0;
}

void MatroskaFile::getH265ConfigData(MatroskaTrack const* track,
                                     u_int8_t*& vps,
                                     unsigned& vpsSize,
                                     u_int8_t*& sps,
                                     unsigned& spsSize,
                                     u_int8_t*& pps,
                                     unsigned& ppsSize) {
    vps = sps = pps = NULL;
    vpsSize = spsSize = ppsSize = 0;

    do {
        if (track == NULL) break;

        u_int8_t* VPS_SPS_PPSBytes = NULL;
        unsigned numVPS_SPS_PPSBytes = 0;
        unsigned i;

        if (track->codecPrivateUsesH264FormatForH265) {
            // The data uses the H.264-style format (but including VPS NAL
            // unit(s)). The VPS,SPS,PPS NAL unit information starts at byte #5:
            if (track->codecPrivateSize >= 6) {
                numVPS_SPS_PPSBytes = track->codecPrivateSize - 5;
                VPS_SPS_PPSBytes = &track->codecPrivate[5];
            }
        } else {
            // The data uses the proper H.265-style format.
            // The VPS,SPS,PPS NAL unit information starts at byte #22:
            if (track->codecPrivateSize >= 23) {
                numVPS_SPS_PPSBytes = track->codecPrivateSize - 22;
                VPS_SPS_PPSBytes = &track->codecPrivate[22];
            }
        }
        if (VPS_SPS_PPSBytes == NULL)
            break;  // no VPS,SPS,PPS NAL unit information was present

        // Extract, from "VPS_SPS_PPSBytes", one VPS NAL unit, one SPS NAL unit,
        // and one PPS NAL unit. (I hope one is all we need of each.)
        u_int8_t* ptr = VPS_SPS_PPSBytes;
        u_int8_t* limit = &VPS_SPS_PPSBytes[numVPS_SPS_PPSBytes];

        if (track->codecPrivateUsesH264FormatForH265) {
            // The data uses the H.264-style format (but including VPS NAL
            // unit(s)).
            while (NUM_BYTES_REMAINING > 0) {
                unsigned numNALUnits = (*ptr++) & 0x1F;
                CHECK_PTR;
                for (i = 0; i < numNALUnits; ++i) {
                    unsigned nalUnitLength = (*ptr++) << 8;
                    CHECK_PTR;
                    nalUnitLength |= *ptr++;
                    CHECK_PTR;

                    if (nalUnitLength > NUM_BYTES_REMAINING) break;
                    u_int8_t nal_unit_type = (ptr[0] & 0x7E) >> 1;
                    if (nal_unit_type == 32) {  // VPS
                        vpsSize = nalUnitLength;
                        delete[] vps;
                        vps = new u_int8_t[nalUnitLength];
                        memmove(vps, ptr, nalUnitLength);
                    } else if (nal_unit_type == 33) {  // SPS
                        spsSize = nalUnitLength;
                        delete[] sps;
                        sps = new u_int8_t[nalUnitLength];
                        memmove(sps, ptr, nalUnitLength);
                    } else if (nal_unit_type == 34) {  // PPS
                        ppsSize = nalUnitLength;
                        delete[] pps;
                        pps = new u_int8_t[nalUnitLength];
                        memmove(pps, ptr, nalUnitLength);
                    }
                    ptr += nalUnitLength;
                }
            }
        } else {
            // The data uses the proper H.265-style format.
            unsigned numOfArrays = *ptr++;
            CHECK_PTR;
            for (unsigned j = 0; j < numOfArrays; ++j) {
                ++ptr;
                CHECK_PTR;  // skip the
                            // 'array_completeness'|'reserved'|'NAL_unit_type'
                            // byte

                unsigned numNalus = (*ptr++) << 8;
                CHECK_PTR;
                numNalus |= *ptr++;
                CHECK_PTR;

                for (i = 0; i < numNalus; ++i) {
                    unsigned nalUnitLength = (*ptr++) << 8;
                    CHECK_PTR;
                    nalUnitLength |= *ptr++;
                    CHECK_PTR;

                    if (nalUnitLength > NUM_BYTES_REMAINING) break;
                    u_int8_t nal_unit_type = (ptr[0] & 0x7E) >> 1;
                    if (nal_unit_type == 32) {  // VPS
                        vpsSize = nalUnitLength;
                        delete[] vps;
                        vps = new u_int8_t[nalUnitLength];
                        memmove(vps, ptr, nalUnitLength);
                    } else if (nal_unit_type == 33) {  // SPS
                        spsSize = nalUnitLength;
                        delete[] sps;
                        sps = new u_int8_t[nalUnitLength];
                        memmove(sps, ptr, nalUnitLength);
                    } else if (nal_unit_type == 34) {  // PPS
                        ppsSize = nalUnitLength;
                        delete[] pps;
                        pps = new u_int8_t[nalUnitLength];
                        memmove(pps, ptr, nalUnitLength);
                    }
                    ptr += nalUnitLength;
                }
            }
        }

        return;
    } while (0);

    // An error occurred:
    delete[] vps;
    vps = NULL;
    vpsSize = 0;
    delete[] sps;
    sps = NULL;
    spsSize = 0;
    delete[] pps;
    pps = NULL;
    ppsSize = 0;
}

void MatroskaFile ::getVorbisOrTheoraConfigData(
        MatroskaTrack const* track,
        u_int8_t*& identificationHeader,
        unsigned& identificationHeaderSize,
        u_int8_t*& commentHeader,
        unsigned& commentHeaderSize,
        u_int8_t*& setupHeader,
        unsigned& setupHeaderSize) {
    identificationHeader = commentHeader = setupHeader = NULL;
    identificationHeaderSize = commentHeaderSize = setupHeaderSize = 0;

    do {
        if (track == NULL) break;

        // The Matroska file's 'Codec Private' data is assumed to be the codec
        // configuration information, containing the "Identification",
        // "Comment", and "Setup" headers. Extract these headers now:
        Boolean isTheora = strcmp(track->mimeType, "video/THEORA") ==
                           0;  // otherwise, Vorbis
        u_int8_t* p = track->codecPrivate;
        unsigned n = track->codecPrivateSize;
        if (n == 0 || p == NULL) break;  // we have no 'Codec Private' data

        u_int8_t numHeaders;
        getPrivByte(numHeaders);
        unsigned headerSize[3];  // we don't handle any more than 2+1 headers

        // Extract the sizes of each of these headers:
        unsigned sizesSum = 0;
        Boolean success = True;
        unsigned i;
        for (i = 0; i < numHeaders && i < 3; ++i) {
            unsigned len = 0;
            u_int8_t c;

            do {
                success = False;
                getPrivByte(c);
                success = True;

                len += c;
            } while (c == 255);
            if (!success || len == 0) break;

            headerSize[i] = len;
            sizesSum += len;
        }
        if (!success) break;

        // Compute the implicit size of the final header:
        if (numHeaders < 3) {
            int finalHeaderSize = n - sizesSum;
            if (finalHeaderSize <= 0) break;  // error in data; give up

            headerSize[numHeaders] = (unsigned)finalHeaderSize;
            ++numHeaders;  // include the final header now
        } else {
            numHeaders = 3;  // The maximum number of headers that we handle
        }

        // Then, extract and classify each header:
        for (i = 0; i < numHeaders; ++i) {
            success = False;
            unsigned newHeaderSize = headerSize[i];
            u_int8_t* newHeader = new u_int8_t[newHeaderSize];
            if (newHeader == NULL) break;

            u_int8_t* hdr = newHeader;
            while (newHeaderSize-- > 0) {
                success = False;
                getPrivByte(*hdr++);
                success = True;
            }
            if (!success) {
                delete[] newHeader;
                break;
            }

            u_int8_t headerType = newHeader[0];
            if (headerType == 1 ||
                (isTheora && headerType == 0x80)) {  // "identification" header
                delete[] identificationHeader;
                identificationHeader = newHeader;
                identificationHeaderSize = headerSize[i];
            } else if (headerType == 3 ||
                       (isTheora && headerType == 0x81)) {  // "comment" header
                delete[] commentHeader;
                commentHeader = newHeader;
                commentHeaderSize = headerSize[i];
            } else if (headerType == 5 ||
                       (isTheora && headerType == 0x82)) {  // "setup" header
                delete[] setupHeader;
                setupHeader = newHeader;
                setupHeaderSize = headerSize[i];
            } else {
                delete[] newHeader;  // because it was a header type that we
                                     // don't understand
            }
        }
        if (!success) break;

        return;
    } while (0);

    // An error occurred:
    delete[] identificationHeader;
    identificationHeader = NULL;
    identificationHeaderSize = 0;
    delete[] commentHeader;
    commentHeader = NULL;
    commentHeaderSize = 0;
    delete[] setupHeader;
    setupHeader = NULL;
    setupHeaderSize = 0;
}

float MatroskaFile::fileDuration() {
    if (fCuePoints == NULL)
        return 0.0;  // Hack, because the RTSP server code assumes that duration
                     // > 0 => seekable. (fix this) #####

    return segmentDuration() * (timecodeScale() / 1000000000.0f);
}

// The size of the largest key frame that we expect.  This determines our buffer
// sizes:
#define MAX_KEY_FRAME_SIZE 300000

FramedSource* MatroskaFile ::createSourceForStreaming(
        FramedSource* baseSource,
        unsigned trackNumber,
        unsigned& estBitrate,
        unsigned& numFiltersInFrontOfTrack) {
    if (baseSource == NULL) return NULL;

    FramedSource* result = baseSource;  // by default
    estBitrate = 100;                   // by default
    numFiltersInFrontOfTrack = 0;       // by default

    // Look at the track's MIME type to set its estimated bitrate (for use by
    // RTCP). (Later, try to be smarter about figuring out the bitrate.) #####
    // Some MIME types also require adding a special 'framer' in front of the
    // source.
    MatroskaTrack* track = lookup(trackNumber);
    if (track != NULL) {  // should always be true
        if (strcmp(track->mimeType, "audio/MPEG") == 0) {
            estBitrate = 128;
        } else if (strcmp(track->mimeType, "audio/AAC") == 0) {
            estBitrate = 96;
        } else if (strcmp(track->mimeType, "audio/AC3") == 0) {
            estBitrate = 48;
        } else if (strcmp(track->mimeType, "audio/VORBIS") == 0) {
            estBitrate = 96;
        } else if (strcmp(track->mimeType, "video/H264") == 0) {
            estBitrate = 500;
            // Allow for the possibility of very large NAL units being fed to
            // the sink object:
            OutPacketBuffer::increaseMaxSizeTo(MAX_KEY_FRAME_SIZE);  // bytes

            // Add a framer in front of the source:
            result = H264VideoStreamDiscreteFramer::createNew(envir(), result);
            ++numFiltersInFrontOfTrack;
        } else if (strcmp(track->mimeType, "video/H265") == 0) {
            estBitrate = 500;
            // Allow for the possibility of very large NAL units being fed to
            // the sink object:
            OutPacketBuffer::increaseMaxSizeTo(MAX_KEY_FRAME_SIZE);  // bytes

            // Add a framer in front of the source:
            result = H265VideoStreamDiscreteFramer::createNew(envir(), result);
            ++numFiltersInFrontOfTrack;
        } else if (strcmp(track->mimeType, "video/VP8") == 0) {
            estBitrate = 500;
        } else if (strcmp(track->mimeType, "video/VP9") == 0) {
            estBitrate = 500;
        } else if (strcmp(track->mimeType, "video/THEORA") == 0) {
            estBitrate = 500;
        } else if (strcmp(track->mimeType, "text/T140") == 0) {
            estBitrate = 48;
        }
    }

    return result;
}

char const* MatroskaFile::trackMIMEType(unsigned trackNumber) const {
    MatroskaTrack* track = lookup(trackNumber);
    if (track == NULL) return NULL;

    return track->mimeType;
}

RTPSink* MatroskaFile ::createRTPSinkForTrackNumber(
        unsigned trackNumber,
        Groupsock* rtpGroupsock,
        unsigned char rtpPayloadTypeIfDynamic) {
    RTPSink* result = NULL;  // default value, if an error occurs

    do {
        MatroskaTrack* track = lookup(trackNumber);
        if (track == NULL) break;

        if (strcmp(track->mimeType, "audio/L16") == 0) {
            result = SimpleRTPSink::createNew(envir(), rtpGroupsock,
                                              rtpPayloadTypeIfDynamic,
                                              track->samplingFrequency, "audio",
                                              "L16", track->numChannels);
        } else if (strcmp(track->mimeType, "audio/MPEG") == 0) {
            result = MPEG1or2AudioRTPSink::createNew(envir(), rtpGroupsock);
        } else if (strcmp(track->mimeType, "audio/AAC") == 0) {
            // The Matroska file's 'Codec Private' data is assumed to be the AAC
            // configuration information.  Use this to generate a hexadecimal
            // 'config' string for the new RTP sink:
            char* configStr = new char[2 * track->codecPrivateSize + 1];
            if (configStr == NULL) break;
            // 2 hex digits per byte, plus the trailing '\0'
            for (unsigned i = 0; i < track->codecPrivateSize; ++i) {
                sprintf(&configStr[2 * i], "%02X", track->codecPrivate[i]);
            }

            result = MPEG4GenericRTPSink::createNew(
                    envir(), rtpGroupsock, rtpPayloadTypeIfDynamic,
                    track->samplingFrequency, "audio", "AAC-hbr", configStr,
                    track->numChannels);
            delete[] configStr;
        } else if (strcmp(track->mimeType, "audio/AC3") == 0) {
            result = AC3AudioRTPSink ::createNew(envir(), rtpGroupsock,
                                                 rtpPayloadTypeIfDynamic,
                                                 track->samplingFrequency);
        } else if (strcmp(track->mimeType, "audio/OPUS") == 0) {
            result = SimpleRTPSink ::createNew(
                    envir(), rtpGroupsock, rtpPayloadTypeIfDynamic, 48000,
                    "audio", "OPUS", 2,
                    False /*only 1 Opus 'packet' in each RTP packet*/);
        } else if (strcmp(track->mimeType, "audio/VORBIS") == 0 ||
                   strcmp(track->mimeType, "video/THEORA") == 0) {
            u_int8_t* identificationHeader;
            unsigned identificationHeaderSize;
            u_int8_t* commentHeader;
            unsigned commentHeaderSize;
            u_int8_t* setupHeader;
            unsigned setupHeaderSize;
            getVorbisOrTheoraConfigData(track, identificationHeader,
                                        identificationHeaderSize, commentHeader,
                                        commentHeaderSize, setupHeader,
                                        setupHeaderSize);

            if (strcmp(track->mimeType, "video/THEORA") == 0) {
                result = TheoraVideoRTPSink ::createNew(
                        envir(), rtpGroupsock, rtpPayloadTypeIfDynamic,
                        identificationHeader, identificationHeaderSize,
                        commentHeader, commentHeaderSize, setupHeader,
                        setupHeaderSize);
            } else {  // Vorbis
                result = VorbisAudioRTPSink ::createNew(
                        envir(), rtpGroupsock, rtpPayloadTypeIfDynamic,
                        track->samplingFrequency, track->numChannels,
                        identificationHeader, identificationHeaderSize,
                        commentHeader, commentHeaderSize, setupHeader,
                        setupHeaderSize);
            }
            delete[] identificationHeader;
            delete[] commentHeader;
            delete[] setupHeader;
        } else if (strcmp(track->mimeType, "video/RAW") == 0) {
            result = RawVideoRTPSink::createNew(
                    envir(), rtpGroupsock, rtpPayloadTypeIfDynamic,
                    track->pixelHeight, track->pixelWidth, track->bitDepth,
                    track->colorSampling, track->colorimetry);
        } else if (strcmp(track->mimeType, "video/H264") == 0) {
            u_int8_t* sps;
            unsigned spsSize;
            u_int8_t* pps;
            unsigned ppsSize;

            getH264ConfigData(track, sps, spsSize, pps, ppsSize);
            result = H264VideoRTPSink::createNew(envir(), rtpGroupsock,
                                                 rtpPayloadTypeIfDynamic, sps,
                                                 spsSize, pps, ppsSize);
            delete[] sps;
            delete[] pps;
        } else if (strcmp(track->mimeType, "video/H265") == 0) {
            u_int8_t* vps;
            unsigned vpsSize;
            u_int8_t* sps;
            unsigned spsSize;
            u_int8_t* pps;
            unsigned ppsSize;

            getH265ConfigData(track, vps, vpsSize, sps, spsSize, pps, ppsSize);
            result = H265VideoRTPSink::createNew(
                    envir(), rtpGroupsock, rtpPayloadTypeIfDynamic, vps,
                    vpsSize, sps, spsSize, pps, ppsSize);
            delete[] vps;
            delete[] sps;
            delete[] pps;
        } else if (strcmp(track->mimeType, "video/VP8") == 0) {
            result = VP8VideoRTPSink::createNew(envir(), rtpGroupsock,
                                                rtpPayloadTypeIfDynamic);
        } else if (strcmp(track->mimeType, "video/VP9") == 0) {
            result = VP9VideoRTPSink::createNew(envir(), rtpGroupsock,
                                                rtpPayloadTypeIfDynamic);
        } else if (strcmp(track->mimeType, "text/T140") == 0) {
            result = T140TextRTPSink::createNew(envir(), rtpGroupsock,
                                                rtpPayloadTypeIfDynamic);
        }
    } while (0);

    return result;
}

FileSink* MatroskaFile::createFileSinkForTrackNumber(unsigned trackNumber,
                                                     char const* fileName) {
    FileSink* result = NULL;            // default value, if an error occurs
    Boolean createOggFileSink = False;  // by default

    do {
        MatroskaTrack* track = lookup(trackNumber);
        if (track == NULL) break;

        if (strcmp(track->mimeType, "video/H264") == 0) {
            u_int8_t* sps;
            unsigned spsSize;
            u_int8_t* pps;
            unsigned ppsSize;

            getH264ConfigData(track, sps, spsSize, pps, ppsSize);

            char* sps_base64 = base64Encode((char*)sps, spsSize);
            char* pps_base64 = base64Encode((char*)pps, ppsSize);
            delete[] sps;
            delete[] pps;

            const char* sPropParameterSetsStr =
                    (std::string(sps_base64) + std::string(sps_base64)).c_str();
            delete[] sps_base64;
            delete[] pps_base64;

            result = H264VideoFileSink::createNew(
                    envir(), fileName, sPropParameterSetsStr,
                    MAX_KEY_FRAME_SIZE);  // extra large buffer size for large
                                          // key frames
        } else if (strcmp(track->mimeType, "video/H265") == 0) {
            u_int8_t* vps;
            unsigned vpsSize;
            u_int8_t* sps;
            unsigned spsSize;
            u_int8_t* pps;
            unsigned ppsSize;

            getH265ConfigData(track, vps, vpsSize, sps, spsSize, pps, ppsSize);

            char* vps_base64 = base64Encode((char*)vps, vpsSize);
            char* sps_base64 = base64Encode((char*)sps, spsSize);
            char* pps_base64 = base64Encode((char*)pps, ppsSize);
            delete[] vps;
            delete[] sps;
            delete[] pps;

            result = H265VideoFileSink::createNew(
                    envir(), fileName, vps_base64, sps_base64, pps_base64,
                    MAX_KEY_FRAME_SIZE);  // extra large buffer size for large
                                          // key frames
            delete[] vps_base64;
            delete[] sps_base64;
            delete[] pps_base64;
        } else if (strcmp(track->mimeType, "video/THEORA") == 0) {
            createOggFileSink = True;
        } else if (strcmp(track->mimeType, "audio/AMR") == 0 ||
                   strcmp(track->mimeType, "audio/AMR-WB") == 0) {
            // For AMR audio streams, we use a special sink that inserts AMR
            // frame hdrs:
            result = AMRAudioFileSink::createNew(envir(), fileName);
        } else if (strcmp(track->mimeType, "audio/VORBIS") == 0 ||
                   strcmp(track->mimeType, "audio/OPUS") == 0) {
            createOggFileSink = True;
        }

        if (createOggFileSink) {
            char* configStr = NULL;  // by default

            if (strcmp(track->mimeType, "audio/VORBIS") == 0 ||
                strcmp(track->mimeType, "video/THEORA") == 0) {
                u_int8_t* identificationHeader;
                unsigned identificationHeaderSize;
                u_int8_t* commentHeader;
                unsigned commentHeaderSize;
                u_int8_t* setupHeader;
                unsigned setupHeaderSize;
                getVorbisOrTheoraConfigData(track, identificationHeader,
                                            identificationHeaderSize,
                                            commentHeader, commentHeaderSize,
                                            setupHeader, setupHeaderSize);
                u_int32_t identField = 0xFACADE;  // Can we get a real value
                                                  // from the file somehow?
                configStr = generateVorbisOrTheoraConfigStr(
                        identificationHeader, identificationHeaderSize,
                        commentHeader, commentHeaderSize, setupHeader,
                        setupHeaderSize, identField);
                delete[] identificationHeader;
                delete[] commentHeader;
                delete[] setupHeader;
            }

            result = OggFileSink::createNew(envir(), fileName,
                                            track->samplingFrequency, configStr,
                                            MAX_KEY_FRAME_SIZE);
            delete[] configStr;
        } else if (result == NULL) {
            // By default, just create a regular "FileSink":
            result = FileSink::createNew(envir(), fileName, MAX_KEY_FRAME_SIZE);
        }
    } while (0);

    return result;
}

void MatroskaFile::addTrack(MatroskaTrack* newTrack, unsigned trackNumber) {
    fTrackTable->add(newTrack, trackNumber);
}

void MatroskaFile::addCuePoint(double cueTime,
                               u_int64_t clusterOffsetInFile,
                               unsigned blockNumWithinCluster) {
    Boolean dummy = False;  // not used
    CuePoint::addCuePoint(fCuePoints, cueTime, clusterOffsetInFile,
                          blockNumWithinCluster, dummy);
}

Boolean MatroskaFile::lookupCuePoint(double& cueTime,
                                     u_int64_t& resultClusterOffsetInFile,
                                     unsigned& resultBlockNumWithinCluster) {
    if (fCuePoints == NULL) return False;

    (void)fCuePoints->lookup(cueTime, resultClusterOffsetInFile,
                             resultBlockNumWithinCluster);
    return True;
}

void MatroskaFile::printCuePoints(FILE* fid) {
    CuePoint::fprintf(fid, fCuePoints);
}

////////// MatroskaTrackTable implementation //////////

MatroskaTrackTable::MatroskaTrackTable()
    : fTable(HashTable::create(ONE_WORD_HASH_KEYS)) {}

MatroskaTrackTable::~MatroskaTrackTable() {
    // Remove and delete all of our "MatroskaTrack" descriptors, and the hash
    // table itself:
    MatroskaTrack* track;
    while ((track = (MatroskaTrack*)fTable->RemoveNext()) != NULL) {
        delete track;
    }
    delete fTable;
}

void MatroskaTrackTable::add(MatroskaTrack* newTrack, unsigned trackNumber) {
    if (newTrack != NULL && newTrack->trackNumber != 0)
        fTable->Remove(reinterpret_cast<char const*>(newTrack->trackNumber));
    MatroskaTrack* existingTrack = (MatroskaTrack*)fTable->Add(
            reinterpret_cast<char const*>(trackNumber), newTrack);
    delete existingTrack;  // in case it wasn't NULL
}

MatroskaTrack* MatroskaTrackTable::lookup(unsigned trackNumber) {
    return (MatroskaTrack*)fTable->Lookup(
            reinterpret_cast<char const*>(trackNumber));
}

unsigned MatroskaTrackTable::numTracks() const { return fTable->numEntries(); }

MatroskaTrackTable::Iterator::Iterator(MatroskaTrackTable& ourTable) {
    fIter = HashTable::Iterator::create(*(ourTable.fTable));
}

MatroskaTrackTable::Iterator::~Iterator() { delete fIter; }

MatroskaTrack* MatroskaTrackTable::Iterator::next() {
    char const* key;
    return (MatroskaTrack*)fIter->next(key);
}

////////// MatroskaTrack implementation //////////

MatroskaTrack::MatroskaTrack()
    : trackNumber(0 /*not set*/),
      trackType(0 /*unknown*/),
      isEnabled(True),
      isDefault(True),
      isForced(False),
      defaultDuration(0),
      name(NULL),
      language(NULL),
      codecID(NULL),
      samplingFrequency(0),
      numChannels(2),
      mimeType(""),
      codecPrivateSize(0),
      codecPrivate(NULL),
      codecPrivateUsesH264FormatForH265(False),
      codecIsOpus(False),
      headerStrippedBytesSize(0),
      headerStrippedBytes(NULL),
      colorSampling(""),
      colorimetry("BT709-2") /*Matroska default value for Primaries */,
      pixelWidth(0),
      pixelHeight(0),
      bitDepth(8),
      subframeSizeSize(0) {}

MatroskaTrack::~MatroskaTrack() {
    delete[] name;
    delete[] language;
    delete[] codecID;
    delete[] codecPrivate;
    delete[] headerStrippedBytes;
}

////////// MatroskaDemux implementation //////////

MatroskaDemux::MatroskaDemux(MatroskaFile& ourFile)
    : Medium(ourFile.envir()),
      fOurFile(ourFile),
      fDemuxedTracksTable(HashTable::create(ONE_WORD_HASH_KEYS)),
      fNextTrackTypeToCheck(0x1) {
    fOurParser = new MatroskaFileParser(
            ourFile,
            ByteStreamFileSource::createNew(envir(), ourFile.fileName()),
            handleEndOfFile, this, this);
}

MatroskaDemux::~MatroskaDemux() {
    // Begin by acting as if we've reached the end of the source file.  This
    // should cause all of our demuxed tracks to get closed.
    handleEndOfFile();

    // Then delete our table of "MatroskaDemuxedTrack"s
    // - but not the "MatroskaDemuxedTrack"s themselves; that should have
    // already happened:
    delete fDemuxedTracksTable;

    delete fOurParser;
    fOurFile.removeDemux(this);
}

FramedSource* MatroskaDemux::newDemuxedTrack() {
    unsigned dummyResultTrackNumber;
    return newDemuxedTrack(dummyResultTrackNumber);
}

FramedSource* MatroskaDemux::newDemuxedTrack(unsigned& resultTrackNumber) {
    FramedSource* result;
    resultTrackNumber = 0;

    for (result = NULL;
         result == NULL && fNextTrackTypeToCheck != MATROSKA_TRACK_TYPE_OTHER;
         fNextTrackTypeToCheck <<= 1) {
        if (fNextTrackTypeToCheck == MATROSKA_TRACK_TYPE_VIDEO)
            resultTrackNumber = fOurFile.chosenVideoTrackNumber();
        else if (fNextTrackTypeToCheck == MATROSKA_TRACK_TYPE_AUDIO)
            resultTrackNumber = fOurFile.chosenAudioTrackNumber();
        else if (fNextTrackTypeToCheck == MATROSKA_TRACK_TYPE_SUBTITLE)
            resultTrackNumber = fOurFile.chosenSubtitleTrackNumber();

        result = newDemuxedTrackByTrackNumber(resultTrackNumber);
    }

    return result;
}

FramedSource* MatroskaDemux::newDemuxedTrackByTrackNumber(
        unsigned trackNumber) {
    if (trackNumber == 0) return NULL;

    FramedSource* trackSource =
            new MatroskaDemuxedTrack(envir(), trackNumber, *this);
    fDemuxedTracksTable->Add(reinterpret_cast<char const*>(trackNumber),
                             trackSource);
    return trackSource;
}

MatroskaDemuxedTrack* MatroskaDemux::lookupDemuxedTrack(unsigned trackNumber) {
    return (MatroskaDemuxedTrack*)fDemuxedTracksTable->Lookup(
            reinterpret_cast<char const*>(trackNumber));
}

void MatroskaDemux::removeTrack(unsigned trackNumber) {
    fDemuxedTracksTable->Remove(reinterpret_cast<char const*>(trackNumber));
    if (fDemuxedTracksTable->numEntries() == 0) {
        // We no longer have any demuxed tracks, so delete ourselves now:
        Medium::close(this);
    }
}

void MatroskaDemux::continueReading() { fOurParser->continueParsing(); }

void MatroskaDemux::seekToTime(double& seekNPT) {
    if (fOurParser != NULL) fOurParser->seekToTime(seekNPT);
}

void MatroskaDemux::handleEndOfFile(void* clientData) {
    ((MatroskaDemux*)clientData)->handleEndOfFile();
}

void MatroskaDemux::handleEndOfFile() {
    // Iterate through all of our 'demuxed tracks', handling 'end of input' on
    // each one. Hack: Because this can cause the hash table to get modified
    // underneath us, we don't call the handlers until after we've first
    // iterated through all of the tracks.
    unsigned numTracks = fDemuxedTracksTable->numEntries();
    if (numTracks == 0) return;
    MatroskaDemuxedTrack** tracks = new MatroskaDemuxedTrack*[numTracks];

    HashTable::Iterator* iter =
            HashTable::Iterator::create(*fDemuxedTracksTable);
    unsigned i;
    char const* trackNumber;

    for (i = 0; i < numTracks; ++i) {
        tracks[i] = (MatroskaDemuxedTrack*)iter->next(trackNumber);
    }
    delete iter;

    for (i = 0; i < numTracks; ++i) {
        if (tracks[i] == NULL) continue;  // sanity check; shouldn't happen
        tracks[i]->handleClosure();
    }

    delete[] tracks;
}

////////// CuePoint implementation //////////

CuePoint::CuePoint(double cueTime,
                   u_int64_t clusterOffsetInFile,
                   unsigned blockNumWithinCluster)
    : fBalance(0),
      fCueTime(cueTime),
      fClusterOffsetInFile(clusterOffsetInFile),
      fBlockNumWithinCluster(blockNumWithinCluster - 1) {
    fSubTree[0] = fSubTree[1] = NULL;
}

CuePoint::~CuePoint() {
    delete fSubTree[0];
    delete fSubTree[1];
}

void CuePoint::addCuePoint(CuePoint*& root,
                           double cueTime,
                           u_int64_t clusterOffsetInFile,
                           unsigned blockNumWithinCluster,
                           Boolean& needToReviseBalanceOfParent) {
    needToReviseBalanceOfParent = False;  // by default; may get changed below

    if (root == NULL) {
        root = new CuePoint(cueTime, clusterOffsetInFile,
                            blockNumWithinCluster);
        needToReviseBalanceOfParent = True;
    } else if (cueTime == root->fCueTime) {
        // Replace existing data:
        root->fClusterOffsetInFile = clusterOffsetInFile;
        root->fBlockNumWithinCluster = blockNumWithinCluster - 1;
    } else {
        // Add to our left or right subtree:
        int direction = cueTime > root->fCueTime;  // 0 (left) or 1 (right)
        Boolean needToReviseOurBalance = False;
        addCuePoint(root->fSubTree[direction], cueTime, clusterOffsetInFile,
                    blockNumWithinCluster, needToReviseOurBalance);

        if (needToReviseOurBalance) {
            // We need to change our 'balance' number, perhaps while also
            // performing a rotation to bring ourself back into balance:
            if (root->fBalance == 0) {
                // We were balanced before, but now we're unbalanced (by 1) on
                // the "direction" side:
                root->fBalance = -1 + 2 * direction;  // -1 for "direction" 0; 1
                                                      // for "direction" 1
                needToReviseBalanceOfParent = True;
            } else if (root->fBalance ==
                       1 - 2 * direction) {  // 1 for "direction" 0; -1 for
                                             // "direction" 1
                // We were unbalanced (by 1) on the side opposite to where we
                // added an entry, so now we're balanced:
                root->fBalance = 0;
            } else {
                // We were unbalanced (by 1) on the side where we added an
                // entry, so now we're unbalanced by 2, and have to rebalance:
                if (root->fSubTree[direction]->fBalance ==
                    -1 + 2 * direction) {  // -1 for "direction" 0; 1 for
                                           // "direction" 1
                    // We're 'doubly-unbalanced' on this side, so perform a
                    // single rotation in the opposite direction:
                    root->fBalance = root->fSubTree[direction]->fBalance = 0;
                    rotate(1 - direction, root);
                } else {
                    // This is the Left-Right case (for "direction" 0) or the
                    // Right-Left case (for "direction" 1); perform two
                    // rotations:
                    char newParentCurBalance = root->fSubTree[direction]
                                                       ->fSubTree[1 - direction]
                                                       ->fBalance;
                    if (newParentCurBalance ==
                        1 - 2 * direction) {  // 1 for "direction" 0; -1 for
                                              // "direction" 1
                        root->fBalance = 0;
                        root->fSubTree[direction]->fBalance =
                                -1 + 2 * direction;  // -1 for "direction" 0; 1
                                                     // for "direction" 1
                    } else if (newParentCurBalance == 0) {
                        root->fBalance = 0;
                        root->fSubTree[direction]->fBalance = 0;
                    } else {
                        root->fBalance =
                                1 - 2 * direction;  // 1 for "direction" 0; -1
                                                    // for "direction" 1
                        root->fSubTree[direction]->fBalance = 0;
                    }
                    rotate(direction, root->fSubTree[direction]);

                    root->fSubTree[direction]->fBalance =
                            0;  // the new root will be balanced
                    rotate(1 - direction, root);
                }
            }
        }
    }
}

Boolean CuePoint::lookup(double& cueTime,
                         u_int64_t& resultClusterOffsetInFile,
                         unsigned& resultBlockNumWithinCluster) {
    if (cueTime < fCueTime) {
        if (left() == NULL) {
            resultClusterOffsetInFile = 0;
            resultBlockNumWithinCluster = 0;
            return False;
        } else {
            return left()->lookup(cueTime, resultClusterOffsetInFile,
                                  resultBlockNumWithinCluster);
        }
    } else {
        if (right() == NULL ||
            !right()->lookup(cueTime, resultClusterOffsetInFile,
                             resultBlockNumWithinCluster)) {
            // Use this record:
            cueTime = fCueTime;
            resultClusterOffsetInFile = fClusterOffsetInFile;
            resultBlockNumWithinCluster = fBlockNumWithinCluster;
        }
        return True;
    }
}

void CuePoint::fprintf(FILE* fid, CuePoint* cuePoint) {
    if (cuePoint != NULL) {
        ::fprintf(fid, "[");
        fprintf(fid, cuePoint->left());

        ::fprintf(fid, ",%.1f{%d},", cuePoint->fCueTime, cuePoint->fBalance);

        fprintf(fid, cuePoint->right());
        ::fprintf(fid, "]");
    }
}

void CuePoint::rotate(unsigned direction /*0 => left; 1 => right*/,
                      CuePoint*& root) {
    CuePoint* pivot = root->fSubTree[1 - direction];  // ASSERT: pivot != NULL
    root->fSubTree[1 - direction] = pivot->fSubTree[direction];
    pivot->fSubTree[direction] = root;
    root = pivot;
}
