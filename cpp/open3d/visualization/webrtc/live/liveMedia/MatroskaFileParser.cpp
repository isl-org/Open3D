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
// A parser for a Matroska file.
// Implementation

#include "MatroskaFileParser.hh"

#include <ByteStreamFileSource.hh>
#include <GroupsockHelper.hh>  // for "gettimeofday()

#include "MatroskaDemuxedTrack.hh"

MatroskaFileParser::MatroskaFileParser(MatroskaFile& ourFile,
                                       FramedSource* inputSource,
                                       FramedSource::onCloseFunc* onEndFunc,
                                       void* onEndClientData,
                                       MatroskaDemux* ourDemux)
    : StreamParser(
              inputSource, onEndFunc, onEndClientData, continueParsing, this),
      fOurFile(ourFile),
      fInputSource(inputSource),
      fOnEndFunc(onEndFunc),
      fOnEndClientData(onEndClientData),
      fOurDemux(ourDemux),
      fCurOffsetInFile(0),
      fSavedCurOffsetInFile(0),
      fLimitOffsetInFile(0),
      fNumHeaderBytesToSkip(0),
      fClusterTimecode(0),
      fBlockTimecode(0),
      fFrameSizesWithinBlock(NULL),
      fPresentationTimeOffset(0.0) {
    if (ourDemux == NULL) {
        // Initialization
        fCurrentParseState = PARSING_START_OF_FILE;

        continueParsing();
    } else {
        fCurrentParseState = LOOKING_FOR_CLUSTER;
        // In this case, parsing (of track data) doesn't start until a client
        // starts reading from a track.
    }
}

MatroskaFileParser::~MatroskaFileParser() {
    delete[] fFrameSizesWithinBlock;
    Medium::close(fInputSource);
}

void MatroskaFileParser::seekToTime(double& seekNPT) {
#ifdef DEBUG
    fprintf(stderr, "seekToTime(%f)\n", seekNPT);
#endif
    if (seekNPT <= 0.0) {
#ifdef DEBUG
        fprintf(stderr, "\t=> start of file\n");
#endif
        seekNPT = 0.0;
        seekToFilePosition(0);
    } else if (seekNPT >= fOurFile.fileDuration()) {
#ifdef DEBUG
        fprintf(stderr, "\t=> end of file\n");
#endif
        seekNPT = fOurFile.fileDuration();
        seekToEndOfFile();
    } else {
        u_int64_t clusterOffsetInFile;
        unsigned blockNumWithinCluster;
        if (!fOurFile.lookupCuePoint(seekNPT, clusterOffsetInFile,
                                     blockNumWithinCluster)) {
#ifdef DEBUG
            fprintf(stderr, "\t=> not supported\n");
#endif
            return;  // seeking not supported
        }

#ifdef DEBUG
        fprintf(stderr,
                "\t=> seek time %f, file position %llu, block number within "
                "cluster %d\n",
                seekNPT, clusterOffsetInFile, blockNumWithinCluster);
#endif
        seekToFilePosition(clusterOffsetInFile);
        fCurrentParseState = LOOKING_FOR_BLOCK;
        // LATER handle "blockNumWithinCluster"; for now, we assume that it's 0
        // #####
    }
}

void MatroskaFileParser ::continueParsing(void* clientData,
                                          unsigned char* /*ptr*/,
                                          unsigned /*size*/,
                                          struct timeval /*presentationTime*/) {
    ((MatroskaFileParser*)clientData)->continueParsing();
}

void MatroskaFileParser::continueParsing() {
    if (fInputSource != NULL) {
        if (!parse()) {
            // We didn't complete the parsing, because we had to read more data
            // from the source, or because we're waiting for another read from
            // downstream.  Once that happens, we'll get called again.
            return;
        }
    }

    // We successfully parsed the file.  Call our 'done' function now:
    if (fOnEndFunc != NULL) (*fOnEndFunc)(fOnEndClientData);
}

Boolean MatroskaFileParser::parse() {
    Boolean areDone = False;

    if (fInputSource->isCurrentlyAwaitingData()) return False;
    // Our input source is currently being read. Wait until that read completes
    try {
        skipRemainingHeaderBytes(True);  // if any
        do {
            if (fInputSource->isCurrentlyAwaitingData()) return False;
            // Our input source is currently being read. Wait until that read
            // completes

            switch (fCurrentParseState) {
                case PARSING_START_OF_FILE: {
                    areDone = parseStartOfFile();
                    break;
                }
                case LOOKING_FOR_TRACKS: {
                    lookForNextTrack();
                    break;
                }
                case PARSING_TRACK: {
                    areDone = parseTrack();
                    if (areDone && fOurFile.fCuesOffset > 0) {
                        // We've finished parsing the 'Track' information. There
                        // are also 'Cues' in the file, so parse those before
                        // finishing: Seek to the specified position in the
                        // file.  We were already told that the 'Cues' begins
                        // there:
#ifdef DEBUG
                        fprintf(stderr,
                                "Seeking to file position %llu (the "
                                "previously-reported location of 'Cues')\n",
                                fOurFile.fCuesOffset);
#endif
                        seekToFilePosition(fOurFile.fCuesOffset);
                        fCurrentParseState = PARSING_CUES;
                        areDone = False;
                    }
                    break;
                }
                case PARSING_CUES: {
                    areDone = parseCues();
                    break;
                }
                case LOOKING_FOR_CLUSTER: {
                    if (fOurFile.fClusterOffset > 0) {
                        // Optimization: Seek to the specified position in the
                        // file.  We were already told that the 'Cluster' begins
                        // there:
#ifdef DEBUG
                        fprintf(stderr,
                                "Optimization: Seeking to file position %llu "
                                "(the previously-reported location of a "
                                "'Cluster')\n",
                                fOurFile.fClusterOffset);
#endif
                        seekToFilePosition(fOurFile.fClusterOffset);
                    }
                    fCurrentParseState = LOOKING_FOR_BLOCK;
                    break;
                }
                case LOOKING_FOR_BLOCK: {
                    lookForNextBlock();
                    break;
                }
                case PARSING_BLOCK: {
                    parseBlock();
                    break;
                }
                case DELIVERING_FRAME_WITHIN_BLOCK: {
                    if (!deliverFrameWithinBlock()) return False;
                    break;
                }
                case DELIVERING_FRAME_BYTES: {
                    deliverFrameBytes();
                    return False;  // Halt parsing for now.  A new 'read' from
                                   // downstream will cause parsing to resume.
                    break;
                }
            }
        } while (!areDone);

        return True;
    } catch (int /*e*/) {
#ifdef DEBUG
        fprintf(stderr,
                "MatroskaFileParser::parse() EXCEPTION (This is normal "
                "behavior - *not* an error)\n");
#endif
        return False;  // the parsing got interrupted
    }
}

Boolean MatroskaFileParser::parseStartOfFile() {
#ifdef DEBUG
    fprintf(stderr, "parsing start of file\n");
#endif
    EBMLId id;
    EBMLDataSize size;

    // The file must begin with the standard EBML header (which we skip):
    if (!parseEBMLIdAndSize(id, size) || id != MATROSKA_ID_EBML) {
        fOurFile.envir() << "ERROR: File does not begin with an EBML header\n";
        return True;  // We're done with the file, because it's not valid
    }
#ifdef DEBUG
    fprintf(stderr,
            "MatroskaFileParser::parseStartOfFile(): Parsed id 0x%s (%s), "
            "size: %lld\n",
            id.hexString(), id.stringName(), size.val());
#endif

    fCurrentParseState = LOOKING_FOR_TRACKS;
    skipHeader(size);

    return False;  // because we have more parsing to do - inside the 'Track'
                   // header
}

void MatroskaFileParser::lookForNextTrack() {
#ifdef DEBUG
    fprintf(stderr, "looking for Track\n");
#endif
    EBMLId id;
    EBMLDataSize size;

    // Read and skip over (or enter) each Matroska header, until we get to a
    // 'Track'.
    while (fCurrentParseState == LOOKING_FOR_TRACKS) {
        while (!parseEBMLIdAndSize(id, size)) {
        }
#ifdef DEBUG
        fprintf(stderr,
                "MatroskaFileParser::lookForNextTrack(): Parsed id 0x%s (%s), "
                "size: %lld\n",
                id.hexString(), id.stringName(), size.val());
#endif
        switch (id.val()) {
            case MATROSKA_ID_SEGMENT: {  // 'Segment' header: enter this
                // Remember the position, within the file, of the start of
                // Segment data, because Seek Positions are relative to this:
                fOurFile.fSegmentDataOffset = fCurOffsetInFile;
                break;
            }
            case MATROSKA_ID_SEEK_HEAD: {  // 'Seek Head' header: enter this
                break;
            }
            case MATROSKA_ID_SEEK: {  // 'Seek' header: enter this
                break;
            }
            case MATROSKA_ID_SEEK_ID: {  // 'Seek ID' header: get this value
                if (parseEBMLNumber(fLastSeekId)) {
#ifdef DEBUG
                    fprintf(stderr, "\tSeek ID 0x%s:\t%s\n",
                            fLastSeekId.hexString(), fLastSeekId.stringName());
#endif
                }
                break;
            }
            case MATROSKA_ID_SEEK_POSITION: {  // 'Seek Position' header: get
                                               // this value
                u_int64_t seekPosition;
                if (parseEBMLVal_unsigned64(size, seekPosition)) {
                    u_int64_t offsetInFile =
                            fOurFile.fSegmentDataOffset + seekPosition;
#ifdef DEBUG
                    fprintf(stderr,
                            "\tSeek Position %llu (=> offset within the file: "
                            "%llu (0x%llx))\n",
                            seekPosition, offsetInFile, offsetInFile);
#endif
                    // The only 'Seek Position's that we care about are for
                    // 'Cluster' and 'Cues':
                    if (fLastSeekId == MATROSKA_ID_CLUSTER) {
                        fOurFile.fClusterOffset = offsetInFile;
                    } else if (fLastSeekId == MATROSKA_ID_CUES) {
                        fOurFile.fCuesOffset = offsetInFile;
                    }
                }
                break;
            }
            case MATROSKA_ID_INFO: {  // 'Segment Info' header: enter this
                break;
            }
            case MATROSKA_ID_TIMECODE_SCALE: {  // 'Timecode Scale' header: get
                                                // this value
                unsigned timecodeScale;
                if (parseEBMLVal_unsigned(size, timecodeScale) &&
                    timecodeScale > 0) {
                    fOurFile.fTimecodeScale = timecodeScale;
#ifdef DEBUG
                    fprintf(stderr,
                            "\tTimecode Scale %u ns (=> Segment Duration == %f "
                            "seconds)\n",
                            fOurFile.timecodeScale(),
                            fOurFile.segmentDuration() *
                                    (fOurFile.fTimecodeScale / 1000000000.0f));
#endif
                }
                break;
            }
            case MATROSKA_ID_DURATION: {  // 'Segment Duration' header: get this
                                          // value
                if (parseEBMLVal_float(size, fOurFile.fSegmentDuration)) {
#ifdef DEBUG
                    fprintf(stderr, "\tSegment Duration %f (== %f seconds)\n",
                            fOurFile.segmentDuration(),
                            fOurFile.segmentDuration() *
                                    (fOurFile.fTimecodeScale / 1000000000.0f));
#endif
                }
                break;
            }
#ifdef DEBUG
            case MATROSKA_ID_TITLE: {  // 'Segment Title': display this value
                char* title;
                if (parseEBMLVal_string(size, title)) {
#ifdef DEBUG
                    fprintf(stderr, "\tTitle: %s\n", title);
#endif
                    delete[] title;
                }
                break;
            }
#endif
            case MATROSKA_ID_TRACKS: {  // enter this, and move on to parsing
                                        // 'Tracks'
                fLimitOffsetInFile = fCurOffsetInFile +
                                     size.val();  // Make sure we don't read
                                                  // past the end of this header
                fCurrentParseState = PARSING_TRACK;
                break;
            }
            default: {  // skip over this header
                skipHeader(size);
                break;
            }
        }
        setParseState();
    }
}

Boolean MatroskaFileParser::parseTrack() {
#ifdef DEBUG
    fprintf(stderr, "parsing Track\n");
#endif
    // Read and process each Matroska header, until we get to the end of the
    // Track:
    MatroskaTrack* track = NULL;
    EBMLId id;
    EBMLDataSize size;
    while (fCurOffsetInFile < fLimitOffsetInFile) {
        while (!parseEBMLIdAndSize(id, size)) {
        }
#ifdef DEBUG
        if (id == MATROSKA_ID_TRACK_ENTRY)
            fprintf(stderr, "\n");  // makes debugging output easier to read
        fprintf(stderr,
                "MatroskaFileParser::parseTrack(): Parsed id 0x%s (%s), size: "
                "%lld\n",
                id.hexString(), id.stringName(), size.val());
#endif
        switch (id.val()) {
            case MATROSKA_ID_TRACK_ENTRY: {  // 'Track Entry' header: enter this
                // Create a new "MatroskaTrack" object for this entry:
                if (track != NULL && track->trackNumber == 0)
                    delete track;  // We had a previous "MatroskaTrack" object
                                   // that was never used
                track = new MatroskaTrack;
                break;
            }
            case MATROSKA_ID_TRACK_NUMBER: {
                unsigned trackNumber;
                if (parseEBMLVal_unsigned(size, trackNumber)) {
#ifdef DEBUG
                    fprintf(stderr, "\tTrack Number %d\n", trackNumber);
#endif
                    if (track != NULL && trackNumber != 0) {
                        track->trackNumber = trackNumber;
                        fOurFile.addTrack(track, trackNumber);
                    }
                }
                break;
            }
            case MATROSKA_ID_TRACK_TYPE: {
                unsigned trackType;
                if (parseEBMLVal_unsigned(size, trackType) && track != NULL) {
                    // We convert the Matroska 'track type' code into our own
                    // code (which we can use as a bitmap):
                    track->trackType =
                            trackType == 1
                                    ? MATROSKA_TRACK_TYPE_VIDEO
                                    : trackType == 2
                                              ? MATROSKA_TRACK_TYPE_AUDIO
                                              : trackType == 0x11
                                                        ? MATROSKA_TRACK_TYPE_SUBTITLE
                                                        : MATROSKA_TRACK_TYPE_OTHER;
#ifdef DEBUG
                    fprintf(stderr, "\tTrack Type 0x%02x (%s)\n", trackType,
                            track->trackType == MATROSKA_TRACK_TYPE_VIDEO
                                    ? "video"
                                    : track->trackType ==
                                                      MATROSKA_TRACK_TYPE_AUDIO
                                              ? "audio"
                                              : track->trackType ==
                                                                MATROSKA_TRACK_TYPE_SUBTITLE
                                                        ? "subtitle"
                                                        : "<other>");
#endif
                }
                break;
            }
            case MATROSKA_ID_FLAG_ENABLED: {
                unsigned flagEnabled;
                if (parseEBMLVal_unsigned(size, flagEnabled)) {
#ifdef DEBUG
                    fprintf(stderr, "\tTrack is Enabled: %d\n", flagEnabled);
#endif
                    if (track != NULL) track->isEnabled = flagEnabled != 0;
                }
                break;
            }
            case MATROSKA_ID_FLAG_DEFAULT: {
                unsigned flagDefault;
                if (parseEBMLVal_unsigned(size, flagDefault)) {
#ifdef DEBUG
                    fprintf(stderr, "\tTrack is Default: %d\n", flagDefault);
#endif
                    if (track != NULL) track->isDefault = flagDefault != 0;
                }
                break;
            }
            case MATROSKA_ID_FLAG_FORCED: {
                unsigned flagForced;
                if (parseEBMLVal_unsigned(size, flagForced)) {
#ifdef DEBUG
                    fprintf(stderr, "\tTrack is Forced: %d\n", flagForced);
#endif
                    if (track != NULL) track->isForced = flagForced != 0;
                }
                break;
            }
            case MATROSKA_ID_DEFAULT_DURATION: {
                unsigned defaultDuration;
                if (parseEBMLVal_unsigned(size, defaultDuration)) {
#ifdef DEBUG
                    fprintf(stderr, "\tDefault duration %f ms\n",
                            defaultDuration / 1000000.0);
#endif
                    if (track != NULL) track->defaultDuration = defaultDuration;
                }
                break;
            }
            case MATROSKA_ID_MAX_BLOCK_ADDITION_ID: {
                unsigned maxBlockAdditionID;
                if (parseEBMLVal_unsigned(size, maxBlockAdditionID)) {
#ifdef DEBUG
                    fprintf(stderr, "\tMax Block Addition ID: %u\n",
                            maxBlockAdditionID);
#endif
                }
                break;
            }
            case MATROSKA_ID_NAME: {
                char* name;
                if (parseEBMLVal_string(size, name)) {
#ifdef DEBUG
                    fprintf(stderr, "\tName: %s\n", name);
#endif
                    if (track != NULL) {
                        delete[] track->name;
                        track->name = name;
                    } else {
                        delete[] name;
                    }
                }
                break;
            }
            case MATROSKA_ID_LANGUAGE: {
                char* language;
                if (parseEBMLVal_string(size, language)) {
#ifdef DEBUG
                    fprintf(stderr, "\tLanguage: %s\n", language);
#endif
                    if (track != NULL) {
                        delete[] track->language;
                        track->language = language;
                    } else {
                        delete[] language;
                    }
                }
                break;
            }
            case MATROSKA_ID_CODEC: {
                char* codecID;
                if (parseEBMLVal_string(size, codecID)) {
#ifdef DEBUG
                    fprintf(stderr, "\tCodec ID: %s\n", codecID);
#endif
                    if (track != NULL) {
                        delete[] track->codecID;
                        track->codecID = codecID;

                        // Also set the track's "mimeType" field, if we can
                        // deduce it from the "codecID":
                        if (strcmp(codecID, "A_PCM/INT/BIG") == 0) {
                            track->mimeType = "audio/L16";
                        } else if (strncmp(codecID, "A_MPEG", 6) == 0) {
                            track->mimeType = "audio/MPEG";
                        } else if (strncmp(codecID, "A_AAC", 5) == 0) {
                            track->mimeType = "audio/AAC";
                        } else if (strncmp(codecID, "A_AC3", 5) == 0) {
                            track->mimeType = "audio/AC3";
                        } else if (strncmp(codecID, "A_VORBIS", 8) == 0) {
                            track->mimeType = "audio/VORBIS";
                        } else if (strcmp(codecID, "A_OPUS") == 0) {
                            track->mimeType = "audio/OPUS";
                            track->codecIsOpus = True;
                        } else if (strcmp(codecID, "V_MPEG4/ISO/AVC") == 0) {
                            track->mimeType = "video/H264";
                        } else if (strcmp(codecID, "V_MPEGH/ISO/HEVC") == 0) {
                            track->mimeType = "video/H265";
                        } else if (strncmp(codecID, "V_VP8", 5) == 0) {
                            track->mimeType = "video/VP8";
                        } else if (strncmp(codecID, "V_VP9", 5) == 0) {
                            track->mimeType = "video/VP9";
                        } else if (strncmp(codecID, "V_THEORA", 8) == 0) {
                            track->mimeType = "video/THEORA";
                        } else if (strncmp(codecID, "S_TEXT", 6) == 0) {
                            track->mimeType = "text/T140";
                        } else if (strncmp(codecID, "V_MJPEG", 7) == 0) {
                            track->mimeType = "video/JPEG";
                        } else if (strncmp(codecID, "V_UNCOMPRESSED", 14) ==
                                   0) {
                            track->mimeType = "video/RAW";
                        }
                    } else {
                        delete[] codecID;
                    }
                }
                break;
            }
            case MATROSKA_ID_CODEC_PRIVATE: {
                u_int8_t* codecPrivate;
                unsigned codecPrivateSize;
                if (parseEBMLVal_binary(size, codecPrivate)) {
                    codecPrivateSize = (unsigned)size.val();
#ifdef DEBUG
                    fprintf(stderr, "\tCodec Private: ");
                    for (unsigned i = 0; i < codecPrivateSize; ++i)
                        fprintf(stderr, "%02x:", codecPrivate[i]);
                    fprintf(stderr, "\n");
#endif
                    if (track != NULL) {
                        delete[] track->codecPrivate;
                        track->codecPrivate = codecPrivate;
                        track->codecPrivateSize = codecPrivateSize;

                        // Hack for H.264 and H.265: The 'codec private' data
                        // contains the size of NAL unit lengths:
                        if (track->codecID != NULL) {
                            if (strcmp(track->codecID, "V_MPEG4/ISO/AVC") ==
                                0) {  // H.264
                                // Byte 4 of the 'codec private' data contains
                                // 'lengthSizeMinusOne':
                                if (codecPrivateSize >= 5)
                                    track->subframeSizeSize =
                                            (codecPrivate[4] & 0x3) + 1;
                            } else if (strcmp(track->codecID,
                                              "V_MPEGH/ISO/HEVC") ==
                                       0) {  // H.265
                                // H.265 'codec private' data is *supposed* to
                                // use the format that's described in
                                // http://lists.matroska.org/pipermail/matroska-devel/2013-September/004567.html
                                // However, some Matroska files use the same
                                // format that was used for H.264. We check for
                                // this here, by checking various fields that
                                // are supposed to be 'all-1' in the 'correct'
                                // format:
                                if (codecPrivateSize < 23 ||
                                    (codecPrivate[13] & 0xF0) != 0xF0 ||
                                    (codecPrivate[15] & 0xFC) != 0xFC ||
                                    (codecPrivate[16] & 0xFC) != 0xFC ||
                                    (codecPrivate[17] & 0xF8) != 0xF8 ||
                                    (codecPrivate[18] & 0xF8) != 0xF8) {
                                    // The 'correct' format isn't being used, so
                                    // assume the H.264 format instead:
                                    track->codecPrivateUsesH264FormatForH265 =
                                            True;

                                    // Byte 4 of the 'codec private' data
                                    // contains 'lengthSizeMinusOne':
                                    if (codecPrivateSize >= 5)
                                        track->subframeSizeSize =
                                                (codecPrivate[4] & 0x3) + 1;
                                } else {
                                    // This looks like the 'correct' format:
                                    track->codecPrivateUsesH264FormatForH265 =
                                            False;

                                    // Byte 21 of the 'codec private' data
                                    // contains 'lengthSizeMinusOne':
                                    track->subframeSizeSize =
                                            (codecPrivate[21] & 0x3) + 1;
                                }
                            }
                        }
                    } else {
                        delete[] codecPrivate;
                    }
                }
                break;
            }
            case MATROSKA_ID_VIDEO: {  // 'Video settings' header: enter this
                break;
            }
            case MATROSKA_ID_PIXEL_WIDTH: {
                unsigned pixelWidth;
                if (parseEBMLVal_unsigned(size, pixelWidth)) {
#ifdef DEBUG
                    fprintf(stderr, "\tPixel Width %d\n", pixelWidth);
#endif
                    if (track != NULL) track->pixelWidth = pixelWidth;
                }
                break;
            }
            case MATROSKA_ID_PIXEL_HEIGHT: {
                unsigned pixelHeight;
                if (parseEBMLVal_unsigned(size, pixelHeight)) {
#ifdef DEBUG
                    fprintf(stderr, "\tPixel Height %d\n", pixelHeight);
#endif
                    if (track != NULL) track->pixelHeight = pixelHeight;
                }
                break;
            }
            case MATROSKA_ID_DISPLAY_WIDTH: {
                unsigned displayWidth;
                if (parseEBMLVal_unsigned(size, displayWidth)) {
#ifdef DEBUG
                    fprintf(stderr, "\tDisplay Width %d\n", displayWidth);
#endif
                }
                break;
            }
            case MATROSKA_ID_DISPLAY_HEIGHT: {
                unsigned displayHeight;
                if (parseEBMLVal_unsigned(size, displayHeight)) {
#ifdef DEBUG
                    fprintf(stderr, "\tDisplay Height %d\n", displayHeight);
#endif
                }
                break;
            }
            case MATROSKA_ID_DISPLAY_UNIT: {
                unsigned displayUnit;
                if (parseEBMLVal_unsigned(size, displayUnit)) {
#ifdef DEBUG
                    fprintf(stderr, "\tDisplay Unit %d\n", displayUnit);
#endif
                }
                break;
            }
            case MATROSKA_ID_AUDIO: {  // 'Audio settings' header: enter this
                break;
            }
            case MATROSKA_ID_SAMPLING_FREQUENCY: {
                float samplingFrequency;
                if (parseEBMLVal_float(size, samplingFrequency)) {
                    if (track != NULL) {
                        track->samplingFrequency = (unsigned)samplingFrequency;
#ifdef DEBUG
                        fprintf(stderr, "\tSampling frequency %f (->%d)\n",
                                samplingFrequency, track->samplingFrequency);
#endif
                    }
                }
                break;
            }
            case MATROSKA_ID_OUTPUT_SAMPLING_FREQUENCY: {
                float outputSamplingFrequency;
                if (parseEBMLVal_float(size, outputSamplingFrequency)) {
#ifdef DEBUG
                    fprintf(stderr, "\tOutput sampling frequency %f\n",
                            outputSamplingFrequency);
#endif
                }
                break;
            }
            case MATROSKA_ID_CHANNELS: {
                unsigned numChannels;
                if (parseEBMLVal_unsigned(size, numChannels)) {
#ifdef DEBUG
                    fprintf(stderr, "\tChannels %d\n", numChannels);
#endif
                    if (track != NULL) track->numChannels = numChannels;
                }
                break;
            }
            case MATROSKA_ID_BIT_DEPTH: {
                unsigned bitDepth;
                if (parseEBMLVal_unsigned(size, bitDepth)) {
#ifdef DEBUG
                    fprintf(stderr, "\tBit Depth %d\n", bitDepth);
#endif
                    if (track != NULL) track->bitDepth = bitDepth;
                }
                break;
            }
            case MATROSKA_ID_CONTENT_ENCODINGS:
            case MATROSKA_ID_CONTENT_ENCODING: {  // 'Content Encodings' or
                                                  // 'Content Encoding' header:
                                                  // enter this
                break;
            }
            case MATROSKA_ID_CONTENT_COMPRESSION: {  // 'Content Compression'
                                                     // header: enter this
                // Note: We currently support only 'Header Stripping'
                // compression, not 'zlib' compression (the default algorithm).
                // Therefore, we disable this track, unless/until we later see
                // that 'Header Stripping' is supported:
                if (track != NULL) track->isEnabled = False;
                break;
            }
            case MATROSKA_ID_CONTENT_COMP_ALGO: {
                unsigned contentCompAlgo;
                if (parseEBMLVal_unsigned(size, contentCompAlgo)) {
#ifdef DEBUG
                    fprintf(stderr, "\tContent Compression Algorithm %d (%s)\n",
                            contentCompAlgo,
                            contentCompAlgo == 0
                                    ? "zlib"
                                    : contentCompAlgo == 3 ? "Header Stripping"
                                                           : "<unknown>");
#endif
                    // The only compression algorithm that we support is #3:
                    // Header Stripping; disable the track otherwise
                    if (track != NULL) track->isEnabled = contentCompAlgo == 3;
                }
                break;
            }
            case MATROSKA_ID_CONTENT_COMP_SETTINGS: {
                u_int8_t* headerStrippedBytes;
                unsigned headerStrippedBytesSize;
                if (parseEBMLVal_binary(size, headerStrippedBytes)) {
                    headerStrippedBytesSize = (unsigned)size.val();
#ifdef DEBUG
                    fprintf(stderr, "\tHeader Stripped Bytes: ");
                    for (unsigned i = 0; i < headerStrippedBytesSize; ++i)
                        fprintf(stderr, "%02x:", headerStrippedBytes[i]);
                    fprintf(stderr, "\n");
#endif
                    if (track != NULL) {
                        delete[] track->headerStrippedBytes;
                        track->headerStrippedBytes = headerStrippedBytes;
                        track->headerStrippedBytesSize =
                                headerStrippedBytesSize;
                    } else {
                        delete[] headerStrippedBytes;
                    }
                }
                break;
            }
            case MATROSKA_ID_CONTENT_ENCRYPTION: {  // 'Content Encrpytion'
                                                    // header: skip this
                // Note: We don't currently support encryption at all.
                // Therefore, we disable this track:
                if (track != NULL) track->isEnabled = False;
                // Fall through to...
            }
            case MATROSKA_ID_COLOR_SPACE: {
                u_int8_t* colourSpace;
                unsigned colourSpaceSize;
                if (parseEBMLVal_binary(size, colourSpace)) {
                    colourSpaceSize = (unsigned)size.val();
#ifdef DEBUG
                    fprintf(stderr, "\tColor space : %02x %02x %02x %02x\n",
                            colourSpace[0], colourSpace[1], colourSpace[2],
                            colourSpace[3]);
#endif
                    if ((track != NULL) && (colourSpaceSize == 4)) {
                        // convert to sampling value (rfc 4175)
                        if ((strncmp((const char*)colourSpace, "I420", 4) ==
                             0) ||
                            (strncmp((const char*)colourSpace, "IYUV", 4) ==
                             0)) {
                            track->colorSampling = "YCbCr-4:2:0";
                        } else if ((strncmp((const char*)colourSpace, "YUY2",
                                            4) == 0) ||
                                   (strncmp((const char*)colourSpace, "UYVY",
                                            4) == 0)) {
                            track->colorSampling = "YCbCr-4:2:2";
                        } else if (strncmp((const char*)colourSpace, "AYUV",
                                           4) == 0) {
                            track->colorSampling = "YCbCr-4:4:4";
                        } else if ((strncmp((const char*)colourSpace, "Y41P",
                                            4) == 0) ||
                                   (strncmp((const char*)colourSpace, "Y41T",
                                            4) == 0)) {
                            track->colorSampling = "YCbCr-4:1:1";
                        } else if (strncmp((const char*)colourSpace, "RGBA",
                                           4) == 0) {
                            track->colorSampling = "RGBA";
                        } else if (strncmp((const char*)colourSpace, "BGRA",
                                           4) == 0) {
                            track->colorSampling = "BGRA";
                        }
                    } else {
                        delete[] colourSpace;
                    }
                }
                break;
            }
            case MATROSKA_ID_PRIMARIES: {
                unsigned primaries;
                if (parseEBMLVal_unsigned(size, primaries)) {
#ifdef DEBUG
                    fprintf(stderr, "\tPrimaries %u\n", primaries);
#endif
                    if (track != NULL) {
                        switch (primaries) {
                            case 1:  // ITU-R BT.709
                                track->colorimetry = "BT709-2";
                                break;
                            case 7:  // SMPTE 240M
                                track->colorimetry = "SMPTE240M";
                                break;
                            case 2:  // Unspecified
                            case 3:  // Reserved
                            case 4:  // ITU-R BT.470M
                            case 5:  // ITU-R BT.470BG
                            case 6:  // SMPTE 170M
                            case 8:  // FILM
                            case 9:  // ITU-R BT.2020
                            default:
#ifdef DEBUG
                                fprintf(stderr,
                                        "\tUnsupported color primaries %u\n",
                                        primaries);
#endif
                                break;
                        }
                    }
                }
            }
            default: {  // We don't process this header, so just skip over it:
                skipHeader(size);
                break;
            }
        }
        setParseState();
    }

    fLimitOffsetInFile = 0;  // reset
    if (track != NULL && track->trackNumber == 0)
        delete track;  // We had a previous "MatroskaTrack" object that was
                       // never used
    return True;       // we're done parsing track entries
}

void MatroskaFileParser::lookForNextBlock() {
#ifdef DEBUG
    fprintf(stderr, "looking for Block\n");
#endif
    // Read and skip over each Matroska header, until we get to a 'Cluster':
    EBMLId id;
    EBMLDataSize size;
    while (fCurrentParseState == LOOKING_FOR_BLOCK) {
        while (!parseEBMLIdAndSize(id, size)) {
        }
#ifdef DEBUG
        fprintf(stderr,
                "MatroskaFileParser::lookForNextBlock(): Parsed id 0x%s (%s), "
                "size: %lld\n",
                id.hexString(), id.stringName(), size.val());
#endif
        switch (id.val()) {
            case MATROSKA_ID_SEGMENT: {  // 'Segment' header: enter this
                break;
            }
            case MATROSKA_ID_CLUSTER: {  // 'Cluster' header: enter this
                break;
            }
            case MATROSKA_ID_TIMECODE: {  // 'Timecode' header: get this value
                unsigned timecode;
                if (parseEBMLVal_unsigned(size, timecode)) {
                    fClusterTimecode = timecode;
#ifdef DEBUG
                    fprintf(stderr, "\tCluster timecode: %d (== %f seconds)\n",
                            fClusterTimecode,
                            fClusterTimecode *
                                    (fOurFile.fTimecodeScale / 1000000000.0));
#endif
                }
                break;
            }
            case MATROSKA_ID_BLOCK_GROUP: {  // 'Block Group' header: enter this
                break;
            }
            case MATROSKA_ID_SIMPLEBLOCK:
            case MATROSKA_ID_BLOCK: {  // 'SimpleBlock' or 'Block' header: enter
                                       // this (and we're done)
                fBlockSize = (unsigned)size.val();
                fCurrentParseState = PARSING_BLOCK;
                break;
            }
            case MATROSKA_ID_BLOCK_DURATION: {  // 'Block Duration' header: get
                                                // this value (but we currently
                                                // don't do anything with it)
                unsigned blockDuration;
                if (parseEBMLVal_unsigned(size, blockDuration)) {
#ifdef DEBUG
                    fprintf(stderr, "\tblock duration: %d (== %f ms)\n",
                            blockDuration,
                            (float)(blockDuration * fOurFile.fTimecodeScale /
                                    1000000.0));
#endif
                }
                break;
            }
            // Attachments are parsed only if we're in DEBUG mode (otherwise we
            // just skip over them):
#ifdef DEBUG
            case MATROSKA_ID_ATTACHMENTS: {  // 'Attachments': enter this
                break;
            }
            case MATROSKA_ID_ATTACHED_FILE: {  // 'Attached File': enter this
                break;
            }
            case MATROSKA_ID_FILE_DESCRIPTION: {  // 'File Description': get
                                                  // this value
                char* fileDescription;
                if (parseEBMLVal_string(size, fileDescription)) {
#ifdef DEBUG
                    fprintf(stderr, "\tFile Description: %s\n",
                            fileDescription);
#endif
                    delete[] fileDescription;
                }
                break;
            }
            case MATROSKA_ID_FILE_NAME: {  // 'File Name': get this value
                char* fileName;
                if (parseEBMLVal_string(size, fileName)) {
#ifdef DEBUG
                    fprintf(stderr, "\tFile Name: %s\n", fileName);
#endif
                    delete[] fileName;
                }
                break;
            }
            case MATROSKA_ID_FILE_MIME_TYPE: {  // 'File MIME Type': get this
                                                // value
                char* fileMIMEType;
                if (parseEBMLVal_string(size, fileMIMEType)) {
#ifdef DEBUG
                    fprintf(stderr, "\tFile MIME Type: %s\n", fileMIMEType);
#endif
                    delete[] fileMIMEType;
                }
                break;
            }
            case MATROSKA_ID_FILE_UID: {  // 'File UID': get this value
                unsigned fileUID;
                if (parseEBMLVal_unsigned(size, fileUID)) {
#ifdef DEBUG
                    fprintf(stderr, "\tFile UID: 0x%x\n", fileUID);
#endif
                }
                break;
            }
#endif
            default: {  // skip over this header
                skipHeader(size);
                break;
            }
        }
        setParseState();
    }
}

Boolean MatroskaFileParser::parseCues() {
#if defined(DEBUG) || defined(DEBUG_CUES)
    fprintf(stderr, "parsing Cues\n");
#endif
    EBMLId id;
    EBMLDataSize size;

    // Read the next header, which should be MATROSKA_ID_CUES:
    if (!parseEBMLIdAndSize(id, size) || id != MATROSKA_ID_CUES)
        return True;  // The header wasn't what we expected, so we're done
    fLimitOffsetInFile =
            fCurOffsetInFile +
            size.val();  // Make sure we don't read past the end of this header

    double currentCueTime = 0.0;
    u_int64_t currentClusterOffsetInFile = 0;

    while (fCurOffsetInFile < fLimitOffsetInFile) {
        while (!parseEBMLIdAndSize(id, size)) {
        }
#ifdef DEBUG_CUES
        if (id == MATROSKA_ID_CUE_POINT)
            fprintf(stderr, "\n");  // makes debugging output easier to read
        fprintf(stderr,
                "MatroskaFileParser::parseCues(): Parsed id 0x%s (%s), size: "
                "%lld\n",
                id.hexString(), id.stringName(), size.val());
#endif
        switch (id.val()) {
            case MATROSKA_ID_CUE_POINT: {  // 'Cue Point' header: enter this
                break;
            }
            case MATROSKA_ID_CUE_TIME: {  // 'Cue Time' header: get this value
                unsigned cueTime;
                if (parseEBMLVal_unsigned(size, cueTime)) {
                    currentCueTime =
                            cueTime * (fOurFile.fTimecodeScale / 1000000000.0);
#ifdef DEBUG_CUES
                    fprintf(stderr, "\tCue Time %d (== %f seconds)\n", cueTime,
                            currentCueTime);
#endif
                }
                break;
            }
            case MATROSKA_ID_CUE_TRACK_POSITIONS: {  // 'Cue Track Positions'
                                                     // header: enter this
                break;
            }
            case MATROSKA_ID_CUE_TRACK: {  // 'Cue Track' header: get this value
                                           // (but only for debugging; we don't
                                           // do anything with it)
                unsigned cueTrack;
                if (parseEBMLVal_unsigned(size, cueTrack)) {
#ifdef DEBUG_CUES
                    fprintf(stderr, "\tCue Track %d\n", cueTrack);
#endif
                }
                break;
            }
            case MATROSKA_ID_CUE_CLUSTER_POSITION: {  // 'Cue Cluster Position'
                                                      // header: get this value
                u_int64_t cueClusterPosition;
                if (parseEBMLVal_unsigned64(size, cueClusterPosition)) {
                    currentClusterOffsetInFile =
                            fOurFile.fSegmentDataOffset + cueClusterPosition;
#ifdef DEBUG_CUES
                    fprintf(stderr,
                            "\tCue Cluster Position %llu (=> offset within the "
                            "file: %llu (0x%llx))\n",
                            cueClusterPosition, currentClusterOffsetInFile,
                            currentClusterOffsetInFile);
#endif
                    // Record this cue point:
                    fOurFile.addCuePoint(
                            currentCueTime, currentClusterOffsetInFile,
                            1 /*default block number within cluster*/);
                }
                break;
            }
            case MATROSKA_ID_CUE_BLOCK_NUMBER: {  // 'Cue Block Number' header:
                                                  // get this value
                unsigned cueBlockNumber;
                if (parseEBMLVal_unsigned(size, cueBlockNumber) &&
                    cueBlockNumber != 0) {
#ifdef DEBUG_CUES
                    fprintf(stderr, "\tCue Block Number %d\n", cueBlockNumber);
#endif
                    // Record this cue point (overwriting any existing entry for
                    // this cue time):
                    fOurFile.addCuePoint(currentCueTime,
                                         currentClusterOffsetInFile,
                                         cueBlockNumber);
                }
                break;
            }
            default: {  // We don't process this header, so just skip over it:
                skipHeader(size);
                break;
            }
        }
        setParseState();
    }

    fLimitOffsetInFile = 0;  // reset
#if defined(DEBUG) || defined(DEBUG_CUES)
    fprintf(stderr, "done parsing Cues\n");
#endif
#ifdef DEBUG_CUES
    fprintf(stderr, "Cue Point tree: ");
    fOurFile.printCuePoints(stderr);
    fprintf(stderr, "\n");
#endif
    return True;  // we're done parsing Cues
}

typedef enum {
    NoLacing,
    XiphLacing,
    FixedSizeLacing,
    EBMLLacing
} MatroskaLacingType;

void MatroskaFileParser::parseBlock() {
#ifdef DEBUG
    fprintf(stderr, "parsing SimpleBlock or Block\n");
#endif
    do {
        unsigned blockStartPos = curOffset();

        // The block begins with the track number:
        EBMLNumber trackNumber;
        if (!parseEBMLNumber(trackNumber)) break;
        fBlockTrackNumber = (unsigned)trackNumber.val();

        // If this track is not being read, then skip the rest of this block,
        // and look for another one:
        if (fOurDemux->lookupDemuxedTrack(fBlockTrackNumber) == NULL) {
            unsigned headerBytesSeen = curOffset() - blockStartPos;
            if (headerBytesSeen < fBlockSize) {
                skipBytes(fBlockSize - headerBytesSeen);
            }
#ifdef DEBUG
            fprintf(stderr, "\tSkipped block for unused track number %d\n",
                    fBlockTrackNumber);
#endif
            fCurrentParseState = LOOKING_FOR_BLOCK;
            setParseState();
            return;
        }

        MatroskaTrack* track = fOurFile.lookup(fBlockTrackNumber);
        if (track == NULL) break;  // shouldn't happen

        // The next two bytes are the block's timecode (relative to the cluster
        // timecode)
        fBlockTimecode = (get1Byte() << 8) | get1Byte();

        // The next byte indicates the type of 'lacing' used:
        u_int8_t c = get1Byte();
        c &= 0x6;  // we're interested in bits 5-6 only
        MatroskaLacingType lacingType =
                (c == 0x0) ? NoLacing
                           : (c == 0x02) ? XiphLacing
                                         : (c == 0x04) ? FixedSizeLacing
                                                       : EBMLLacing;
#ifdef DEBUG
        fprintf(stderr,
                "\ttrack number %d, timecode %d (=> %f seconds), %s lacing\n",
                fBlockTrackNumber, fBlockTimecode,
                (fClusterTimecode + fBlockTimecode) *
                        (fOurFile.fTimecodeScale / 1000000000.0),
                (lacingType == NoLacing)
                        ? "no"
                        : (lacingType == XiphLacing)
                                  ? "Xiph"
                                  : (lacingType == FixedSizeLacing)
                                            ? "fixed-size"
                                            : "EBML");
#endif

        if (lacingType == NoLacing) {
            fNumFramesInBlock = 1;
        } else {
            // The next byte tells us how many frames are present in this block
            fNumFramesInBlock = get1Byte() + 1;
        }
        delete[] fFrameSizesWithinBlock;
        fFrameSizesWithinBlock = new unsigned[fNumFramesInBlock];
        if (fFrameSizesWithinBlock == NULL) break;

        if (lacingType == NoLacing) {
            unsigned headerBytesSeen = curOffset() - blockStartPos;
            if (headerBytesSeen > fBlockSize) break;

            fFrameSizesWithinBlock[0] = fBlockSize - headerBytesSeen;
        } else if (lacingType == FixedSizeLacing) {
            unsigned headerBytesSeen = curOffset() - blockStartPos;
            if (headerBytesSeen > fBlockSize) break;

            unsigned frameBytesAvailable = fBlockSize - headerBytesSeen;
            unsigned constantFrameSize =
                    frameBytesAvailable / fNumFramesInBlock;

            for (unsigned i = 0; i < fNumFramesInBlock; ++i) {
                fFrameSizesWithinBlock[i] = constantFrameSize;
            }
            // If there are any bytes left over, assign them to the last frame:
            fFrameSizesWithinBlock[fNumFramesInBlock - 1] +=
                    frameBytesAvailable % fNumFramesInBlock;
        } else {  // EBML or Xiph lacing
            unsigned curFrameSize = 0;
            unsigned frameSizesTotal = 0;
            unsigned i;

            for (i = 0; i < fNumFramesInBlock - 1; ++i) {
                if (lacingType == EBMLLacing) {
                    EBMLNumber frameSize;
                    if (!parseEBMLNumber(frameSize)) break;
                    unsigned fsv = (unsigned)frameSize.val();

                    if (i == 0) {
                        curFrameSize = fsv;
                    } else {
                        // The value we read is a signed value, that's added to
                        // the previous frame size, to get the current frame
                        // size:
                        unsigned toSubtract =
                                (fsv > 0xFFFFFF)
                                        ? 0x07FFFFFF
                                        : (fsv > 0xFFFF) ? 0x0FFFFF
                                                         : (fsv > 0xFF) ? 0x1FFF
                                                                        : 0x3F;
                        int fsv_signed = fsv - toSubtract;
                        curFrameSize += fsv_signed;
                        if ((int)curFrameSize < 0) break;
                    }
                } else {  // Xiph lacing
                    curFrameSize = 0;
                    u_int8_t c;
                    do {
                        c = get1Byte();
                        curFrameSize += c;
                    } while (c == 0xFF);
                }
                fFrameSizesWithinBlock[i] = curFrameSize;
                frameSizesTotal += curFrameSize;
            }
            if (i != fNumFramesInBlock - 1)
                break;  // an error occurred within the "for" loop

            // Compute the size of the final frame within the block (from the
            // block's size, and the frame sizes already computed):)
            unsigned headerBytesSeen = curOffset() - blockStartPos;
            if (headerBytesSeen + frameSizesTotal > fBlockSize) break;
            fFrameSizesWithinBlock[i] =
                    fBlockSize - (headerBytesSeen + frameSizesTotal);
        }

        // We're done parsing headers within the block, and (as a result) we now
        // know the sizes of all frames within the block. If we have 'stripped
        // bytes' that are common to (the front of) all frames, then count them
        // now:
        if (track->headerStrippedBytesSize != 0) {
            for (unsigned i = 0; i < fNumFramesInBlock; ++i)
                fFrameSizesWithinBlock[i] += track->headerStrippedBytesSize;
        }
#ifdef DEBUG
        fprintf(stderr, "\tThis block contains %d frame(s); size(s):",
                fNumFramesInBlock);
        unsigned frameSizesTotal = 0;
        for (unsigned i = 0; i < fNumFramesInBlock; ++i) {
            fprintf(stderr, " %d", fFrameSizesWithinBlock[i]);
            frameSizesTotal += fFrameSizesWithinBlock[i];
        }
        if (fNumFramesInBlock > 1)
            fprintf(stderr, " (total: %u)", frameSizesTotal);
        fprintf(stderr, " bytes\n");
#endif
        // Next, start delivering these frames:
        fCurrentParseState = DELIVERING_FRAME_WITHIN_BLOCK;
        fCurOffsetWithinFrame = fNextFrameNumberToDeliver = 0;
        setParseState();
        return;
    } while (0);

    // An error occurred.  Try to recover:
#ifdef DEBUG
    fprintf(stderr, "parseBlock(): Error parsing data; trying to recover...\n");
#endif
    fCurrentParseState = LOOKING_FOR_BLOCK;
}

Boolean MatroskaFileParser::deliverFrameWithinBlock() {
#ifdef DEBUG
    fprintf(stderr, "delivering frame within SimpleBlock or Block\n");
#endif
    do {
        MatroskaTrack* track = fOurFile.lookup(fBlockTrackNumber);
        if (track == NULL) break;  // shouldn't happen

        MatroskaDemuxedTrack* demuxedTrack =
                fOurDemux->lookupDemuxedTrack(fBlockTrackNumber);
        if (demuxedTrack == NULL) break;  // shouldn't happen
        if (!demuxedTrack->isCurrentlyAwaitingData()) {
            // Someone has been reading this stream, but isn't right now.
            // We can't deliver this frame until he asks for it, so punt for
            // now. The next time he asks for a frame, he'll get it.
#ifdef DEBUG
            fprintf(stderr, "\tdeferring delivery of frame #%d (%d bytes)",
                    fNextFrameNumberToDeliver,
                    fFrameSizesWithinBlock[fNextFrameNumberToDeliver]);
            if (track->haveSubframes())
                fprintf(stderr, "[offset %d]", fCurOffsetWithinFrame);
            fprintf(stderr, "\n");
#endif
            restoreSavedParserState();  // so we read from the beginning next
                                        // time
            return False;
        }

        unsigned frameSize;
        u_int8_t const* specialFrameSource = NULL;
        u_int8_t const opusCommentHeader[16] = {
                'O', 'p', 'u', 's', 'T', 'a', 'g', 's', 0, 0, 0, 0, 0, 0, 0, 0};
        if (track->codecIsOpus && demuxedTrack->fOpusTrackNumber < 2) {
            // Special case for Opus audio.  The first frame (the
            // 'configuration' header) comes from the 'private data'.  The
            // second frame (the 'comment' header) comes is synthesized by us
            // here:
            if (demuxedTrack->fOpusTrackNumber == 0) {
                specialFrameSource = track->codecPrivate;
                frameSize = track->codecPrivateSize;
            } else {  // demuxedTrack->fOpusTrackNumber == 1
                specialFrameSource = opusCommentHeader;
                frameSize = sizeof opusCommentHeader;
            }
            ++demuxedTrack->fOpusTrackNumber;
        } else {
            frameSize = fFrameSizesWithinBlock[fNextFrameNumberToDeliver];
            if (track->haveSubframes()) {
                // The next "track->subframeSizeSize" bytes contain the length
                // of a 'subframe':
                if (fCurOffsetWithinFrame + track->subframeSizeSize > frameSize)
                    break;  // sanity check
                unsigned subframeSize = 0;
                for (unsigned i = 0; i < track->subframeSizeSize; ++i) {
                    u_int8_t c;
                    getCommonFrameBytes(track, &c, 1, 0);
                    if (fCurFrameNumBytesToGet > 0) {  // it'll be 1
                        c = get1Byte();
                        ++fCurOffsetWithinFrame;
                    }
                    subframeSize = subframeSize * 256 + c;
                }
                if (subframeSize == 0 ||
                    fCurOffsetWithinFrame + subframeSize > frameSize)
                    break;  // sanity check
                frameSize = subframeSize;
            }
        }

        // Compute the presentation time of this frame (from the cluster
        // timecode, the block timecode, and the default duration):
        double pt = (fClusterTimecode + fBlockTimecode) *
                            (fOurFile.fTimecodeScale / 1000000000.0) +
                    fNextFrameNumberToDeliver *
                            (track->defaultDuration / 1000000000.0);
        if (fPresentationTimeOffset == 0.0) {
            // This is the first time we've computed a presentation time.
            // Compute an offset to make the presentation times aligned with
            // 'wall clock' time:
            struct timeval timeNow;
            gettimeofday(&timeNow, NULL);
            double ptNow = timeNow.tv_sec + timeNow.tv_usec / 1000000.0;
            fPresentationTimeOffset = ptNow - pt;
        }
        pt += fPresentationTimeOffset;
        struct timeval presentationTime;
        presentationTime.tv_sec = (unsigned)pt;
        presentationTime.tv_usec =
                (unsigned)((pt - presentationTime.tv_sec) * 1000000);
        unsigned durationInMicroseconds;
        if (specialFrameSource != NULL) {
            durationInMicroseconds = 0;
        } else {  // normal case
            durationInMicroseconds = track->defaultDuration / 1000;
            if (track->haveSubframes()) {
                // If this is a 'subframe', use a duration of 0 instead (unless
                // it's the last 'subframe'):
                if (fCurOffsetWithinFrame + frameSize +
                            track->subframeSizeSize <
                    fFrameSizesWithinBlock[fNextFrameNumberToDeliver]) {
                    // There's room for at least one more subframe after this,
                    // so give this subframe a duration of 0
                    durationInMicroseconds = 0;
                }
            }
        }

        if (track->defaultDuration == 0) {
            // Adjust the frame duration to keep the sum of frame durations
            // aligned with presentation times.
            if (demuxedTrack->prevPresentationTime().tv_sec !=
                0) {  // not the first time for this track
                demuxedTrack->durationImbalance() +=
                        (presentationTime.tv_sec -
                         demuxedTrack->prevPresentationTime().tv_sec) *
                                1000000 +
                        (presentationTime.tv_usec -
                         demuxedTrack->prevPresentationTime().tv_usec);
            }
            int adjustment = 0;
            if (demuxedTrack->durationImbalance() > 0) {
                // The duration needs to be increased.
                int const adjustmentThreshold =
                        100000;  // don't increase the duration by more than
                                 // this amount (in case there's a mistake)
                adjustment =
                        demuxedTrack->durationImbalance() > adjustmentThreshold
                                ? adjustmentThreshold
                                : demuxedTrack->durationImbalance();
            } else if (demuxedTrack->durationImbalance() < 0) {
                // The duration needs to be decreased.
                adjustment = (unsigned)(-demuxedTrack->durationImbalance()) <
                                             durationInMicroseconds
                                     ? demuxedTrack->durationImbalance()
                                     : -(int)durationInMicroseconds;
            }
            durationInMicroseconds += adjustment;
            demuxedTrack->durationImbalance() -=
                    durationInMicroseconds;  // for next time
            demuxedTrack->prevPresentationTime() =
                    presentationTime;  // for next time
        }

        demuxedTrack->presentationTime() = presentationTime;
        demuxedTrack->durationInMicroseconds() = durationInMicroseconds;

        // Deliver the next block now:
        if (frameSize > demuxedTrack->maxSize()) {
            demuxedTrack->numTruncatedBytes() =
                    frameSize - demuxedTrack->maxSize();
            demuxedTrack->frameSize() = demuxedTrack->maxSize();
        } else {  // normal case
            demuxedTrack->numTruncatedBytes() = 0;
            demuxedTrack->frameSize() = frameSize;
        }
        getCommonFrameBytes(track, demuxedTrack->to(),
                            demuxedTrack->frameSize(),
                            demuxedTrack->numTruncatedBytes());

        // Next, deliver (and/or skip) bytes from the input file:
        if (specialFrameSource != NULL) {
            memmove(demuxedTrack->to(), specialFrameSource,
                    demuxedTrack->frameSize());
#ifdef DEBUG
            fprintf(stderr, "\tdelivered special frame: %d bytes",
                    demuxedTrack->frameSize());
            if (demuxedTrack->numTruncatedBytes() > 0)
                fprintf(stderr, " (%d bytes truncated)",
                        demuxedTrack->numTruncatedBytes());
            fprintf(stderr, " @%u.%06u (%.06f from start); duration %u us\n",
                    demuxedTrack->presentationTime().tv_sec,
                    demuxedTrack->presentationTime().tv_usec,
                    demuxedTrack->presentationTime().tv_sec +
                            demuxedTrack->presentationTime().tv_usec /
                                    1000000.0 -
                            fPresentationTimeOffset,
                    demuxedTrack->durationInMicroseconds());
#endif
            setParseState();
            FramedSource::afterGetting(demuxedTrack);  // completes delivery
        } else {                                       // normal case
            fCurrentParseState = DELIVERING_FRAME_BYTES;
            setParseState();
        }
        return True;
    } while (0);

    // An error occurred.  Try to recover:
#ifdef DEBUG
    fprintf(stderr,
            "deliverFrameWithinBlock(): Error parsing data; trying to "
            "recover...\n");
#endif
    fCurrentParseState = LOOKING_FOR_BLOCK;
    return True;
}

void MatroskaFileParser::deliverFrameBytes() {
    do {
        MatroskaTrack* track = fOurFile.lookup(fBlockTrackNumber);
        if (track == NULL) break;  // shouldn't happen

        MatroskaDemuxedTrack* demuxedTrack =
                fOurDemux->lookupDemuxedTrack(fBlockTrackNumber);
        if (demuxedTrack == NULL) break;  // shouldn't happen

        unsigned const BANK_SIZE = bankSize();
        while (fCurFrameNumBytesToGet > 0) {
            // Hack: We can get no more than BANK_SIZE bytes at a time:
            unsigned numBytesToGet = fCurFrameNumBytesToGet > BANK_SIZE
                                             ? BANK_SIZE
                                             : fCurFrameNumBytesToGet;
            getBytes(fCurFrameTo, numBytesToGet);
            fCurFrameTo += numBytesToGet;
            fCurFrameNumBytesToGet -= numBytesToGet;
            fCurOffsetWithinFrame += numBytesToGet;
            setParseState();
        }
        while (fCurFrameNumBytesToSkip > 0) {
            // Hack: We can skip no more than BANK_SIZE bytes at a time:
            unsigned numBytesToSkip = fCurFrameNumBytesToSkip > BANK_SIZE
                                              ? BANK_SIZE
                                              : fCurFrameNumBytesToSkip;
            skipBytes(numBytesToSkip);
            fCurFrameNumBytesToSkip -= numBytesToSkip;
            fCurOffsetWithinFrame += numBytesToSkip;
            setParseState();
        }
#ifdef DEBUG
        fprintf(stderr, "\tdelivered frame #%d: %d bytes",
                fNextFrameNumberToDeliver, demuxedTrack->frameSize());
        if (track->haveSubframes())
            fprintf(stderr, "[offset %d]",
                    fCurOffsetWithinFrame - track->subframeSizeSize -
                            demuxedTrack->frameSize() -
                            demuxedTrack->numTruncatedBytes());
        if (demuxedTrack->numTruncatedBytes() > 0)
            fprintf(stderr, " (%d bytes truncated)",
                    demuxedTrack->numTruncatedBytes());
        fprintf(stderr, " @%u.%06u (%.06f from start); duration %u us\n",
                demuxedTrack->presentationTime().tv_sec,
                demuxedTrack->presentationTime().tv_usec,
                demuxedTrack->presentationTime().tv_sec +
                        demuxedTrack->presentationTime().tv_usec / 1000000.0 -
                        fPresentationTimeOffset,
                demuxedTrack->durationInMicroseconds());
#endif

        if (!track->haveSubframes() ||
            fCurOffsetWithinFrame + track->subframeSizeSize >=
                    fFrameSizesWithinBlock[fNextFrameNumberToDeliver]) {
            // Either we don't have subframes, or there's no more room for
            // another subframe => We're completely done with this frame now:
            ++fNextFrameNumberToDeliver;
            fCurOffsetWithinFrame = 0;
        }
        if (fNextFrameNumberToDeliver == fNumFramesInBlock) {
            // We've delivered all of the frames from this block.  Look for
            // another block next:
            fCurrentParseState = LOOKING_FOR_BLOCK;
        } else {
            fCurrentParseState = DELIVERING_FRAME_WITHIN_BLOCK;
        }

        setParseState();
        FramedSource::afterGetting(demuxedTrack);  // completes delivery
        return;
    } while (0);

    // An error occurred.  Try to recover:
#ifdef DEBUG
    fprintf(stderr,
            "deliverFrameBytes(): Error parsing data; trying to recover...\n");
#endif
    fCurrentParseState = LOOKING_FOR_BLOCK;
}

void MatroskaFileParser ::getCommonFrameBytes(MatroskaTrack* track,
                                              u_int8_t* to,
                                              unsigned numBytesToGet,
                                              unsigned numBytesToSkip) {
    if (track->headerStrippedBytesSize > fCurOffsetWithinFrame) {
        // We have some common 'header stripped' bytes that remain to be
        // prepended to the frame.  Use these first:
        unsigned numRemainingHeaderStrippedBytes =
                track->headerStrippedBytesSize - fCurOffsetWithinFrame;
        unsigned numHeaderStrippedBytesToGet;
        if (numBytesToGet <= numRemainingHeaderStrippedBytes) {
            numHeaderStrippedBytesToGet = numBytesToGet;
            numBytesToGet = 0;
            if (numBytesToGet + numBytesToSkip <=
                numRemainingHeaderStrippedBytes) {
                numBytesToSkip = 0;
            } else {
                numBytesToSkip = numBytesToGet + numBytesToSkip -
                                 numRemainingHeaderStrippedBytes;
            }
        } else {
            numHeaderStrippedBytesToGet = numRemainingHeaderStrippedBytes;
            numBytesToGet = numBytesToGet - numRemainingHeaderStrippedBytes;
        }

        if (numHeaderStrippedBytesToGet > 0) {
            memmove(to, &track->headerStrippedBytes[fCurOffsetWithinFrame],
                    numHeaderStrippedBytesToGet);
            to += numHeaderStrippedBytesToGet;
            fCurOffsetWithinFrame += numHeaderStrippedBytesToGet;
        }
    }

    fCurFrameTo = to;
    fCurFrameNumBytesToGet = numBytesToGet;
    fCurFrameNumBytesToSkip = numBytesToSkip;
}

Boolean MatroskaFileParser::parseEBMLNumber(EBMLNumber& num) {
    unsigned i;
    u_int8_t bitmask = 0x80;
    for (i = 0; i < EBML_NUMBER_MAX_LEN; ++i) {
        while (1) {
            if (fLimitOffsetInFile > 0 && fCurOffsetInFile > fLimitOffsetInFile)
                return False;  // We've hit our pre-set limit
            num.data[i] = get1Byte();
            ++fCurOffsetInFile;

            // If we're looking for an id, skip any leading bytes that don't
            // contain a '1' in the first 4 bits:
            if (i == 0 /*we're a leading byte*/ &&
                !num.stripLeading1 /*we're looking for an id*/ &&
                (num.data[i] & 0xF0) == 0) {
                setParseState();  // ensures that we make forward progress if
                                  // the parsing gets interrupted
                continue;
            }
            break;
        }
        if ((num.data[0] & bitmask) != 0) {
            // num[i] is the last byte of the id
            if (num.stripLeading1) num.data[0] &= ~bitmask;
            break;
        }
        bitmask >>= 1;
    }
    if (i == EBML_NUMBER_MAX_LEN) return False;

    num.len = i + 1;
    return True;
}

Boolean MatroskaFileParser::parseEBMLIdAndSize(EBMLId& id, EBMLDataSize& size) {
    return parseEBMLNumber(id) && parseEBMLNumber(size);
}

Boolean MatroskaFileParser::parseEBMLVal_unsigned64(EBMLDataSize& size,
                                                    u_int64_t& result) {
    u_int64_t sv = size.val();
    if (sv > 8) return False;  // size too large

    result = 0;  // initially
    for (unsigned i = (unsigned)sv; i > 0; --i) {
        if (fLimitOffsetInFile > 0 && fCurOffsetInFile > fLimitOffsetInFile)
            return False;  // We've hit our pre-set limit

        u_int8_t c = get1Byte();
        ++fCurOffsetInFile;

        result = result * 256 + c;
    }

    return True;
}

Boolean MatroskaFileParser::parseEBMLVal_unsigned(EBMLDataSize& size,
                                                  unsigned& result) {
    if (size.val() > 4) return False;  // size too large

    u_int64_t result64;
    if (!parseEBMLVal_unsigned64(size, result64)) return False;

    result = (unsigned)result64;

    return True;
}

Boolean MatroskaFileParser::parseEBMLVal_float(EBMLDataSize& size,
                                               float& result) {
    if (size.val() == 4) {
        // Normal case.  Read the value as if it were a 4-byte integer, then
        // copy it to the 'float' result:
        unsigned resultAsUnsigned;
        if (!parseEBMLVal_unsigned(size, resultAsUnsigned)) return False;

        if (sizeof result != sizeof resultAsUnsigned) return False;
        memcpy(&result, &resultAsUnsigned, sizeof result);
        return True;
    } else if (size.val() == 8) {
        // Read the value as if it were an 8-byte integer, then copy it to a
        // 'double', the convert that to the 'float' result:
        u_int64_t resultAsUnsigned64;
        if (!parseEBMLVal_unsigned64(size, resultAsUnsigned64)) return False;

        double resultDouble;
        if (sizeof resultDouble != sizeof resultAsUnsigned64) return False;
        memcpy(&resultDouble, &resultAsUnsigned64, sizeof resultDouble);

        result = (float)resultDouble;
        return True;
    } else {
        // Unworkable size
        return False;
    }
}

Boolean MatroskaFileParser::parseEBMLVal_string(EBMLDataSize& size,
                                                char*& result) {
    unsigned resultLength = (unsigned)size.val();
    result = new char[resultLength + 1];  // allow for the trailing '\0'
    if (result == NULL) return False;

    char* p = result;
    unsigned i;
    for (i = 0; i < resultLength; ++i) {
        if (fLimitOffsetInFile > 0 && fCurOffsetInFile > fLimitOffsetInFile)
            break;  // We've hit our pre-set limit

        u_int8_t c = get1Byte();
        ++fCurOffsetInFile;

        *p++ = c;
    }
    if (i < resultLength) {  // an error occurred
        delete[] result;
        result = NULL;
        return False;
    }
    *p = '\0';

    return True;
}

Boolean MatroskaFileParser::parseEBMLVal_binary(EBMLDataSize& size,
                                                u_int8_t*& result) {
    unsigned resultLength = (unsigned)size.val();
    result = new u_int8_t[resultLength];
    if (result == NULL) return False;

    u_int8_t* p = result;
    unsigned i;
    for (i = 0; i < resultLength; ++i) {
        if (fLimitOffsetInFile > 0 && fCurOffsetInFile > fLimitOffsetInFile)
            break;  // We've hit our pre-set limit

        u_int8_t c = get1Byte();
        ++fCurOffsetInFile;

        *p++ = c;
    }
    if (i < resultLength) {  // an error occurred
        delete[] result;
        result = NULL;
        return False;
    }

    return True;
}

void MatroskaFileParser::skipHeader(EBMLDataSize const& size) {
    u_int64_t sv = (unsigned)size.val();
#ifdef DEBUG
    fprintf(stderr, "\tskipping %llu bytes\n", sv);
#endif

    fNumHeaderBytesToSkip = sv;
    skipRemainingHeaderBytes(False);
}

void MatroskaFileParser::skipRemainingHeaderBytes(Boolean isContinuation) {
    if (fNumHeaderBytesToSkip == 0) return;  // common case

    // Hack: To avoid tripping into a parser 'internal error' if we try to skip
    // an excessively large distance, break up the skipping into manageable
    // chunks, to ensure forward progress:
    unsigned const maxBytesToSkip = bankSize();
    while (fNumHeaderBytesToSkip > 0) {
        unsigned numBytesToSkipNow = fNumHeaderBytesToSkip < maxBytesToSkip
                                             ? (unsigned)fNumHeaderBytesToSkip
                                             : maxBytesToSkip;
        setParseState();
        skipBytes(numBytesToSkipNow);
#ifdef DEBUG
        if (isContinuation || numBytesToSkipNow < fNumHeaderBytesToSkip) {
            fprintf(stderr, "\t\t(skipped %u bytes; %llu bytes remaining)\n",
                    numBytesToSkipNow,
                    fNumHeaderBytesToSkip - numBytesToSkipNow);
        }
#endif
        fCurOffsetInFile += numBytesToSkipNow;
        fNumHeaderBytesToSkip -= numBytesToSkipNow;
    }
}

void MatroskaFileParser::setParseState() {
    fSavedCurOffsetInFile = fCurOffsetInFile;
    fSavedCurOffsetWithinFrame = fCurOffsetWithinFrame;
    saveParserState();
}

void MatroskaFileParser::restoreSavedParserState() {
    StreamParser::restoreSavedParserState();
    fCurOffsetInFile = fSavedCurOffsetInFile;
    fCurOffsetWithinFrame = fSavedCurOffsetWithinFrame;
}

void MatroskaFileParser::seekToFilePosition(u_int64_t offsetInFile) {
    ByteStreamFileSource* fileSource = (ByteStreamFileSource*)
            fInputSource;  // we know it's a "ByteStreamFileSource"
    if (fileSource != NULL) {
        fileSource->seekToByteAbsolute(offsetInFile);
        resetStateAfterSeeking();
    }
}

void MatroskaFileParser::seekToEndOfFile() {
    ByteStreamFileSource* fileSource = (ByteStreamFileSource*)
            fInputSource;  // we know it's a "ByteStreamFileSource"
    if (fileSource != NULL) {
        fileSource->seekToEnd();
        resetStateAfterSeeking();
    }
}

void MatroskaFileParser::resetStateAfterSeeking() {
    // Because we're resuming parsing after seeking to a new position in the
    // file, reset the parser state:
    fCurOffsetInFile = fSavedCurOffsetInFile = 0;
    fCurOffsetWithinFrame = fSavedCurOffsetWithinFrame = 0;
    flushInput();
}
