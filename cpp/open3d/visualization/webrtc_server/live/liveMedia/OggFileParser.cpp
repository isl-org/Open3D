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
// A parser for an Ogg file.
// Implementation

#include "OggFileParser.hh"

#include <GroupsockHelper.hh>  // for "gettimeofday()

#include "OggDemuxedTrack.hh"

PacketSizeTable::PacketSizeTable(unsigned number_page_segments)
    : numCompletedPackets(0),
      totSizes(0),
      nextPacketNumToDeliver(0),
      lastPacketIsIncomplete(False) {
    size = new unsigned[number_page_segments];
    for (unsigned i = 0; i < number_page_segments; ++i) size[i] = 0;
}

PacketSizeTable::~PacketSizeTable() { delete[] size; }

OggFileParser::OggFileParser(OggFile& ourFile,
                             FramedSource* inputSource,
                             FramedSource::onCloseFunc* onEndFunc,
                             void* onEndClientData,
                             OggDemux* ourDemux)
    : StreamParser(
              inputSource, onEndFunc, onEndClientData, continueParsing, this),
      fOurFile(ourFile),
      fInputSource(inputSource),
      fOnEndFunc(onEndFunc),
      fOnEndClientData(onEndClientData),
      fOurDemux(ourDemux),
      fNumUnfulfilledTracks(0),
      fPacketSizeTable(NULL),
      fCurrentTrackNumber(0),
      fSavedPacket(NULL) {
    if (ourDemux == NULL) {
        // Initialization
        fCurrentParseState = PARSING_START_OF_FILE;
        continueParsing();
    } else {
        fCurrentParseState = PARSING_AND_DELIVERING_PAGES;
        // In this case, parsing (of page data) doesn't start until a client
        // starts reading from a track.
    }
}

OggFileParser::~OggFileParser() {
    delete[] fSavedPacket;
    delete fPacketSizeTable;
    Medium::close(fInputSource);
}

void OggFileParser::continueParsing(void* clientData,
                                    unsigned char* ptr,
                                    unsigned size,
                                    struct timeval presentationTime) {
    ((OggFileParser*)clientData)->continueParsing();
}

void OggFileParser::continueParsing() {
    if (fInputSource != NULL) {
        if (fInputSource->isCurrentlyAwaitingData()) return;
        // Our input source is currently being read. Wait until that read
        // completes

        if (!parse()) {
            // We didn't complete the parsing, because we had to read more data
            // from the source, or because we're waiting for another read from
            // downstream. Once that happens, we'll get called again.
            return;
        }
    }

    // We successfully parsed the file.  Call our 'done' function now:
    if (fOnEndFunc != NULL) (*fOnEndFunc)(fOnEndClientData);
}

Boolean OggFileParser::parse() {
    try {
        while (1) {
            switch (fCurrentParseState) {
                case PARSING_START_OF_FILE: {
                    if (parseStartOfFile()) return True;
                }
                case PARSING_AND_DELIVERING_PAGES: {
                    parseAndDeliverPages();
                }
                case DELIVERING_PACKET_WITHIN_PAGE: {
                    if (deliverPacketWithinPage()) return False;
                }
            }
        }
    } catch (int /*e*/) {
#ifdef DEBUG
        fprintf(stderr,
                "OggFileParser::parse() EXCEPTION (This is normal behavior - "
                "*not* an error)\n");
#endif
        return False;  // the parsing got interrupted
    }
}

Boolean OggFileParser::parseStartOfFile() {
#ifdef DEBUG
    fprintf(stderr, "parsing start of file\n");
#endif
    // Read and parse each 'page', until we see the first non-BOS page, or until
    // we have collected all required headers for Vorbis, Theora, or Opus
    // track(s) (if any).
    u_int8_t header_type_flag;
    do {
        header_type_flag = parseInitialPage();
    } while ((header_type_flag & 0x02) != 0 || needHeaders());

#ifdef DEBUG
    fprintf(stderr, "Finished parsing start of file\n");
#endif
    return True;
}

static u_int32_t byteSwap(u_int32_t x) {
    return (x << 24) | ((x << 8) & 0x00FF0000) | ((x >> 8) & 0x0000FF00) |
           (x >> 24);
}

u_int8_t OggFileParser::parseInitialPage() {
    u_int8_t header_type_flag;
    u_int32_t bitstream_serial_number;
    parseStartOfPage(header_type_flag, bitstream_serial_number);

    // If this is a BOS page, examine the first 8 bytes of the first 'packet',
    // to see whether the track data type is one that we know how to stream:
    OggTrack* track;
    if ((header_type_flag & 0x02) != 0) {  // BOS
        char const* mimeType = NULL;       // if unknown
        if (fPacketSizeTable != NULL &&
            fPacketSizeTable->size[0] >= 8) {  // sanity check
            char buf[8];
            testBytes((u_int8_t*)buf, 8);

            if (strncmp(&buf[1], "vorbis", 6) == 0) {
                mimeType = "audio/VORBIS";
                ++fNumUnfulfilledTracks;
            } else if (strncmp(buf, "OpusHead", 8) == 0) {
                mimeType = "audio/OPUS";
                ++fNumUnfulfilledTracks;
            } else if (strncmp(&buf[1], "theora", 6) == 0) {
                mimeType = "video/THEORA";
                ++fNumUnfulfilledTracks;
            }
        }

        // Add a new track descriptor for this track:
        track = new OggTrack;
        track->trackNumber = bitstream_serial_number;
        track->mimeType = mimeType;
        fOurFile.addTrack(track);
    } else {  // not a BOS page
        // Because this is not a BOS page, the specified track should already
        // have been seen:
        track = fOurFile.lookup(bitstream_serial_number);
    }

    if (track != NULL) {  // sanity check
#ifdef DEBUG
        fprintf(stderr, "This track's MIME type: %s\n",
                track->mimeType == NULL ? "(unknown)" : track->mimeType);
#endif
        if (track->mimeType != NULL &&
            (strcmp(track->mimeType, "audio/VORBIS") == 0 ||
             strcmp(track->mimeType, "video/THEORA") == 0 ||
             strcmp(track->mimeType, "audio/OPUS") == 0)) {
            // Special-case handling of Vorbis, Theora, or Opus tracks:
            // Make a copy of each packet, until we get the three special
            // headers that we need:
            Boolean isVorbis = strcmp(track->mimeType, "audio/VORBIS") == 0;
            Boolean isTheora = strcmp(track->mimeType, "video/THEORA") == 0;

            for (unsigned j = 0; j < fPacketSizeTable->numCompletedPackets &&
                                 track->weNeedHeaders();
                 ++j) {
                unsigned const packetSize = fPacketSizeTable->size[j];
                if (packetSize == 0) continue;  // sanity check

                delete[] fSavedPacket /*if any*/;
                fSavedPacket = new u_int8_t[packetSize];
                getBytes(fSavedPacket, packetSize);
                fPacketSizeTable->totSizes -= packetSize;

                // The start of the packet tells us whether its a header that we
                // know about:
                Boolean headerIsKnown = False;
                unsigned index = 0;
                if (isVorbis) {
                    u_int8_t const firstByte = fSavedPacket[0];

                    headerIsKnown =
                            firstByte == 1 || firstByte == 3 || firstByte == 5;
                    index = (firstByte - 1) / 2;  // 1, 3, or 5 => 0, 1, or 2
                } else if (isTheora) {
                    u_int8_t const firstByte = fSavedPacket[0];

                    headerIsKnown = firstByte == 0x80 || firstByte == 0x81 ||
                                    firstByte == 0x82;
                    index = firstByte &
                            ~0x80;  // 0x80, 0x81, or 0x82 => 0, 1, or 2
                } else {            // Opus
                    if (strncmp((char const*)fSavedPacket, "OpusHead", 8) ==
                        0) {
                        headerIsKnown = True;
                        index = 0;  // "identification" header
                    } else if (strncmp((char const*)fSavedPacket, "OpusTags",
                                       8) == 0) {
                        headerIsKnown = True;
                        index = 1;  // "comment" header
                    }
                }
                if (headerIsKnown) {
#ifdef DEBUG
                    char const* headerName[3] = {"identification", "comment",
                                                 "setup"};
                    fprintf(stderr, "Saved %d-byte %s \"%s\" header\n",
                            packetSize, track->mimeType, headerName[index]);
#endif
                    // This is a header, but first check it for validity:
                    if (!validateHeader(track, fSavedPacket, packetSize))
                        continue;

                    // Save this header (deleting any old header of the same
                    // type that we'd saved before)
                    delete[] track->vtoHdrs.header[index];
                    track->vtoHdrs.header[index] = fSavedPacket;
                    fSavedPacket = NULL;
                    track->vtoHdrs.headerSize[index] = packetSize;

                    if (!track->weNeedHeaders()) {
                        // We now have all of the needed Vorbis, Theora, or Opus
                        // headers for this track:
                        --fNumUnfulfilledTracks;
                    }
                    // Note: The above code won't work if a required header is
                    // fragmented over more than one 'page'.  We assume that
                    // that won't ever happen...
                }
            }
        }
    }

    // Skip over any remaining packet data bytes:
    if (fPacketSizeTable->totSizes > 0) {
#ifdef DEBUG
        fprintf(stderr, "Skipping %d remaining packet data bytes\n",
                fPacketSizeTable->totSizes);
#endif
        skipBytes(fPacketSizeTable->totSizes);
    }

    return header_type_flag;
}

// A simple bit vector class for reading bits in little-endian order.
// (We can't use our usual "BitVector" class, because that's big-endian.)
class LEBitVector {
public:
    LEBitVector(u_int8_t const* p, unsigned numBytes)
        : fPtr(p), fEnd(&p[numBytes]), fNumBitsRemainingInCurrentByte(8) {}

    u_int32_t getBits(unsigned numBits /*<=32*/) {
        if (noMoreBits()) {
            return 0;
        } else if (numBits == fNumBitsRemainingInCurrentByte) {
            u_int32_t result =
                    (*fPtr++) >> (8 - fNumBitsRemainingInCurrentByte);
            fNumBitsRemainingInCurrentByte = 8;

            return result;
        } else if (numBits < fNumBitsRemainingInCurrentByte) {
            u_int8_t mask = 0xFF >> (8 - numBits);
            u_int32_t result =
                    ((*fPtr) >> (8 - fNumBitsRemainingInCurrentByte)) & mask;
            fNumBitsRemainingInCurrentByte -= numBits;

            return result;
        } else {  // numBits > fNumBitsRemainingInCurrentByte
            // Do two recursive calls to get the result:
            unsigned nbr = fNumBitsRemainingInCurrentByte;
            u_int32_t firstBits = getBits(nbr);
            u_int32_t nextBits = getBits(numBits - nbr);

            return (nextBits << nbr) | firstBits;
        }
    }

    void skipBits(unsigned numBits) {
        while (numBits > 32) {
            (void)getBits(32);
            numBits -= 32;
        }
        (void)getBits(numBits);
    }

    unsigned numBitsRemaining() {
        return (fEnd - fPtr - 1) * 8 + fNumBitsRemainingInCurrentByte;
    }
    Boolean noMoreBits() const { return fPtr >= fEnd; }

private:
    u_int8_t const* fPtr;
    u_int8_t const* fEnd;
    unsigned fNumBitsRemainingInCurrentByte;  // 1..8
};

static unsigned ilog(int n) {
    if (n < 0) return 0;

    unsigned x = (unsigned)n;
    unsigned result = 0;

    while (x > 0) {
        ++result;
        x >>= 1;
    }

    return result;
}

static unsigned lookup1_values(unsigned codebook_entries,
                               unsigned codebook_dimensions) {
    // "the greatest integer value for which [return_value] to the power of
    // [codebook_dimensions]
    //  is less than or equal to [codebook_entries]"
    unsigned return_value = 0;
    unsigned powerValue;

    do {
        ++return_value;
        // Compute powerValue = return_value ** codebook_dimensions
        if (return_value == 1)
            powerValue = 1;  // optimization
        else {
            powerValue = 1;
            for (unsigned i = 0; i < codebook_dimensions; ++i) {
                powerValue *= return_value;
            }
        }
    } while (powerValue <= codebook_entries);
    return_value -= 1;

    return return_value;
}

static Boolean parseVorbisSetup_codebook(LEBitVector& bv) {
    if (bv.noMoreBits()) return False;

    unsigned sync = bv.getBits(24);
    if (sync != 0x564342) return False;
    unsigned codebook_dimensions = bv.getBits(16);
    unsigned codebook_entries = bv.getBits(24);
    unsigned ordered = bv.getBits(1);
#ifdef DEBUG_SETUP_HEADER
    fprintf(stderr,
            "\t\t\tcodebook_dimensions: %d; codebook_entries: %d, ordered: "
            "%d\n",
            codebook_dimensions, codebook_entries, ordered);
#endif
    if (!ordered) {
        unsigned sparse = bv.getBits(1);
#ifdef DEBUG_SETUP_HEADER
        fprintf(stderr, "\t\t\t!ordered: sparse %d\n", sparse);
#endif
        for (unsigned i = 0; i < codebook_entries; ++i) {
            unsigned codewordLength;

            if (sparse) {
                unsigned flag = bv.getBits(1);
                if (flag) {
                    codewordLength = bv.getBits(5) + 1;
                } else {
                    codewordLength = 0;
                }
            } else {
                codewordLength = bv.getBits(5) + 1;
            }
#ifdef DEBUG_SETUP_HEADER
            fprintf(stderr, "\t\t\t\tcodeword length[%d]:\t%d\n", i,
                    codewordLength);
#else
            codewordLength = codewordLength + 0;  // to prevent compiler warning
#endif
        }
    } else {  // ordered
#ifdef DEBUG_SETUP_HEADER
        fprintf(stderr, "\t\t\tordered:\n");
#endif
        unsigned current_entry = 0;
        unsigned current_length = bv.getBits(5) + 1;
        do {
            unsigned number =
                    bv.getBits(ilog(codebook_entries - current_entry));
#ifdef DEBUG_SETUP_HEADER
            fprintf(stderr, "\t\t\t\tcodeword length[%d..%d]:\t%d\n",
                    current_entry, current_entry + number - 1, current_length);
#endif
            current_entry += number;
            if (current_entry > codebook_entries) {
                fprintf(stderr,
                        "Vorbis codebook parsing error: current_entry %d > "
                        "codebook_entries %d!\n",
                        current_entry, codebook_entries);
                return False;
            }
            ++current_length;
        } while (current_entry < codebook_entries);
    }

    unsigned codebook_lookup_type = bv.getBits(4);
#ifdef DEBUG_SETUP_HEADER
    fprintf(stderr, "\t\t\tcodebook_lookup_type: %d\n", codebook_lookup_type);
#endif
    if (codebook_lookup_type > 2) {
        fprintf(stderr,
                "Vorbis codebook parsing error: codebook_lookup_type %d!\n",
                codebook_lookup_type);
        return False;
    } else if (codebook_lookup_type > 0) {  // 1 or 2
        bv.skipBits(32 +
                    32);  // "codebook_minimum_value" and "codebook_delta_value"
        unsigned codebook_value_bits = bv.getBits(4) + 1;
        bv.skipBits(1);  // "codebook_lookup_p"
        unsigned codebook_lookup_values;
        if (codebook_lookup_type == 1) {
            codebook_lookup_values =
                    lookup1_values(codebook_entries, codebook_dimensions);
        } else {  // 2
            codebook_lookup_values = codebook_entries * codebook_dimensions;
        }

        bv.skipBits(codebook_lookup_values *
                    codebook_value_bits);  // "codebook_multiplicands"
    }

    return True;
}

static Boolean parseVorbisSetup_codebooks(LEBitVector& bv) {
    if (bv.noMoreBits()) return False;

    unsigned vorbis_codebook_count = bv.getBits(8) + 1;
#ifdef DEBUG_SETUP_HEADER
    fprintf(stderr, "\tCodebooks: vorbis_codebook_count: %d\n",
            vorbis_codebook_count);
#endif
    for (unsigned i = 0; i < vorbis_codebook_count; ++i) {
#ifdef DEBUG_SETUP_HEADER
        fprintf(stderr, "\t\tCodebook %d:\n", i);
#endif
        if (!parseVorbisSetup_codebook(bv)) return False;
    }

    return True;
}

static Boolean parseVorbisSetup_timeDomainTransforms(LEBitVector& bv) {
    if (bv.noMoreBits()) return False;

    unsigned vorbis_time_count = bv.getBits(6) + 1;
#ifdef DEBUG_SETUP_HEADER
    fprintf(stderr, "\tTime domain transforms: vorbis_time_count: %d\n",
            vorbis_time_count);
#endif
    for (unsigned i = 0; i < vorbis_time_count; ++i) {
        unsigned val = bv.getBits(16);
        if (val != 0) {
            fprintf(stderr,
                    "Vorbis Time domain transforms, read non-zero value %d\n",
                    val);
            return False;
        }
    }

    return True;
}

static Boolean parseVorbisSetup_floors(LEBitVector& bv) {
    if (bv.noMoreBits()) return False;

    unsigned vorbis_floor_count = bv.getBits(6) + 1;
#ifdef DEBUG_SETUP_HEADER
    fprintf(stderr, "\tFloors: vorbis_floor_count: %d\n", vorbis_floor_count);
#endif
    for (unsigned i = 0; i < vorbis_floor_count; ++i) {
        unsigned floorType = bv.getBits(16);
        if (floorType == 0) {
            bv.skipBits(8 + 16 + 16 + 6 + 8);
            unsigned floor0_number_of_books = bv.getBits(4) + 1;
            bv.skipBits(floor0_number_of_books * 8);
        } else if (floorType == 1) {
            unsigned floor1_partitions = bv.getBits(5);

            unsigned* floor1_partition_class_list =
                    new unsigned[floor1_partitions];
            unsigned maximum_class = 0, j;
            for (j = 0; j < floor1_partitions; ++j) {
                floor1_partition_class_list[j] = bv.getBits(4);
                if (floor1_partition_class_list[j] > maximum_class)
                    maximum_class = floor1_partition_class_list[j];
            }

            unsigned* floor1_class_dimensions = new unsigned[maximum_class + 1];
            for (j = 0; j <= maximum_class; ++j) {
                floor1_class_dimensions[j] = bv.getBits(3) + 1;
                unsigned floor1_class_subclasses = bv.getBits(2);
                if (floor1_class_subclasses != 0) {
                    bv.skipBits(8);  // "floor1_class_masterbooks[j]"
                }

                unsigned twoExp_floor1_class_subclasses =
                        1 << floor1_class_subclasses;
                bv.skipBits(twoExp_floor1_class_subclasses *
                            8);  // "floor1_subclass_books[j][*]"
            }

            bv.skipBits(2);  // "floor1_multiplier"
            unsigned rangebits = bv.getBits(4);
            for (j = 0; j < floor1_partitions; ++j) {
                unsigned current_class_number = floor1_partition_class_list[j];
                bv.skipBits(floor1_class_dimensions[current_class_number] *
                            rangebits);
            }

            delete[] floor1_partition_class_list;
            delete[] floor1_class_dimensions;
        } else {  // floorType > 1
            fprintf(stderr, "Vorbis Floors, read bad floor type %d\n",
                    floorType);
            return False;
        }
    }

    return True;
}

static Boolean parseVorbisSetup_residues(LEBitVector& bv) {
    if (bv.noMoreBits()) return False;

    unsigned vorbis_residue_count = bv.getBits(6) + 1;
#ifdef DEBUG_SETUP_HEADER
    fprintf(stderr, "\tResidues: vorbis_residue_count: %d\n",
            vorbis_residue_count);
#endif
    for (unsigned i = 0; i < vorbis_residue_count; ++i) {
        unsigned vorbis_residue_type = bv.getBits(16);
        if (vorbis_residue_type > 2) {
            fprintf(stderr,
                    "Vorbis Residues, read bad vorbis_residue_type: %d\n",
                    vorbis_residue_type);
            return False;
        } else {
            bv.skipBits(24 + 24 + 24);  // "residue_begin", "residue_end",
                                        // "residue_partition_size"
            unsigned residue_classifications = bv.getBits(6) + 1;
            bv.skipBits(8);  // "residue_classbook"

            u_int8_t* residue_cascade = new u_int8_t[residue_classifications];
            unsigned j;
            for (j = 0; j < residue_classifications; ++j) {
                u_int8_t high_bits = 0;
                u_int8_t low_bits = bv.getBits(3);
                unsigned bitflag = bv.getBits(1);
                if (bitflag) {
                    high_bits = bv.getBits(5);
                }

                residue_cascade[j] = (high_bits << 3) | low_bits;
            }

            for (j = 0; j < residue_classifications; ++j) {
                u_int8_t const cascade = residue_cascade[j];
                u_int8_t mask = 0x80;
                while (mask != 0) {
                    if ((cascade & mask) != 0)
                        bv.skipBits(8);  // "residue_books[j][*]"
                    mask >>= 1;
                }
            }

            delete[] residue_cascade;
        }
    }

    return True;
}

static Boolean parseVorbisSetup_mappings(LEBitVector& bv,
                                         unsigned audio_channels) {
    if (bv.noMoreBits()) return False;

    unsigned vorbis_mapping_count = bv.getBits(6) + 1;
#ifdef DEBUG_SETUP_HEADER
    fprintf(stderr, "\tMappings: vorbis_mapping_count: %d\n",
            vorbis_mapping_count);
#endif
    for (unsigned i = 0; i < vorbis_mapping_count; ++i) {
        unsigned vorbis_mapping_type = bv.getBits(16);
        if (vorbis_mapping_type != 0) {
            fprintf(stderr,
                    "Vorbis Mappings, read bad vorbis_mapping_type: %d\n",
                    vorbis_mapping_type);
            return False;
        }

        unsigned vorbis_mapping_submaps = 1;
        if (bv.getBits(1)) vorbis_mapping_submaps = bv.getBits(4) + 1;

        if (bv.getBits(1)) {  // "square polar channel mapping is in use"
            unsigned vorbis_mapping_coupling_steps = bv.getBits(8) + 1;

            for (unsigned j = 0; j < vorbis_mapping_coupling_steps; ++j) {
                unsigned ilog_audio_channels_minus_1 = ilog(audio_channels - 1);
                bv.skipBits(
                        2 *
                        ilog_audio_channels_minus_1);  // "vorbis_mapping_magnitude",
                                                       // "vorbis_mapping_angle"
            }
        }

        unsigned reserved = bv.getBits(2);
        if (reserved != 0) {
            fprintf(stderr, "Vorbis Mappings, read bad 'reserved' field\n");
            return False;
        }

        if (vorbis_mapping_submaps > 1) {
            for (unsigned j = 0; j < audio_channels; ++j) {
                unsigned vorbis_mapping_mux = bv.getBits(4);

                fprintf(stderr, "\t\t\t\tvorbis_mapping_mux[%d]: %d\n", j,
                        vorbis_mapping_mux);
                if (vorbis_mapping_mux >= vorbis_mapping_submaps) {
                    fprintf(stderr,
                            "Vorbis Mappings, read bad \"vorbis_mapping_mux\" "
                            "%d (>= \"vorbis_mapping_submaps\" %d)\n",
                            vorbis_mapping_mux, vorbis_mapping_submaps);
                    return False;
                }
            }
        }

        bv.skipBits(vorbis_mapping_submaps *
                    (8 + 8 + 8));  // "the floor and residue numbers"
    }

    return True;
}

static Boolean parseVorbisSetup_modes(LEBitVector& bv, OggTrack* track) {
    if (bv.noMoreBits()) return False;

    unsigned vorbis_mode_count = bv.getBits(6) + 1;
    unsigned ilog_vorbis_mode_count_minus_1 = ilog(vorbis_mode_count - 1);
#ifdef DEBUG_SETUP_HEADER
    fprintf(stderr, "\tModes: vorbis_mode_count: %d (ilog(%d-1):%d)\n",
            vorbis_mode_count, vorbis_mode_count,
            ilog_vorbis_mode_count_minus_1);
#endif
    track->vtoHdrs.vorbis_mode_count = vorbis_mode_count;
    track->vtoHdrs.ilog_vorbis_mode_count_minus_1 =
            ilog_vorbis_mode_count_minus_1;
    track->vtoHdrs.vorbis_mode_blockflag = new u_int8_t[vorbis_mode_count];

    for (unsigned i = 0; i < vorbis_mode_count; ++i) {
        track->vtoHdrs.vorbis_mode_blockflag[i] = (u_int8_t)bv.getBits(1);
#ifdef DEBUG_SETUP_HEADER
        fprintf(stderr, "\t\tMode %d: vorbis_mode_blockflag: %d\n", i,
                track->vtoHdrs.vorbis_mode_blockflag[i]);
#endif
        bv.skipBits(16 + 16 +
                    8);  // "vorbis_mode_windowtype",
                         // "vorbis_mode_transformtype", "vorbis_mode_mapping"
    }

    return True;
}

static Boolean parseVorbisSetupHeader(OggTrack* track,
                                      u_int8_t const* p,
                                      unsigned headerSize) {
    LEBitVector bv(p, headerSize);
    do {
        if (!parseVorbisSetup_codebooks(bv)) break;
        if (!parseVorbisSetup_timeDomainTransforms(bv)) break;
        if (!parseVorbisSetup_floors(bv)) break;
        if (!parseVorbisSetup_residues(bv)) break;
        if (!parseVorbisSetup_mappings(bv, track->numChannels)) break;
        if (!parseVorbisSetup_modes(bv, track)) break;
        unsigned framingFlag = bv.getBits(1);
        if (framingFlag == 0) {
            fprintf(stderr,
                    "Vorbis \"setup\" header did not end with a 'framing "
                    "flag'!\n");
            break;
        }

        return True;
    } while (0);

    // An error occurred:
    return False;
}

#ifdef DEBUG
#define CHECK_PTR \
    if (p >= pEnd) return False
#define printComment(p, len)                 \
    do {                                     \
        for (unsigned k = 0; k < len; ++k) { \
            CHECK_PTR;                       \
            fprintf(stderr, "%c", *p++);     \
        }                                    \
    } while (0)
#endif

static Boolean validateCommentHeader(u_int8_t const* p,
                                     unsigned headerSize,
                                     unsigned isOpus = 0) {
    if (headerSize < 15 + isOpus) {  // need 7+isOpus + 4(vendor_length) +
                                     // 4(user_comment_list_length)
        fprintf(stderr, "\"comment\" header is too short (%d bytes)\n",
                headerSize);
        return False;
    }

#ifdef DEBUG
    u_int8_t const* pEnd = &p[headerSize];
    p += 7 + isOpus;

    u_int32_t vendor_length = (p[3] << 24) | (p[2] << 16) | (p[1] << 8) | p[0];
    p += 4;
    fprintf(stderr, "\tvendor_string:");
    printComment(p, vendor_length);
    fprintf(stderr, "\n");

    u_int32_t user_comment_list_length =
            (p[3] << 24) | (p[2] << 16) | (p[1] << 8) | p[0];
    p += 4;
    for (unsigned i = 0; i < user_comment_list_length; ++i) {
        CHECK_PTR;
        u_int32_t length = (p[3] << 24) | (p[2] << 16) | (p[1] << 8) | p[0];
        p += 4;
        fprintf(stderr, "\tuser_comment[%d]:", i);
        printComment(p, length);
        fprintf(stderr, "\n");
    }
#endif

    return True;
}

static unsigned blocksizeFromExponent(unsigned exponent) {
    unsigned result = 1;
    for (unsigned i = 0; i < exponent; ++i) result = 2 * result;
    return result;
}

Boolean OggFileParser::validateHeader(OggTrack* track,
                                      u_int8_t const* p,
                                      unsigned headerSize) {
    // Assert: headerSize >= 7 (because we've already checked
    // "<packet_type>XXXXXX" or "OpusXXXX")
    if (strcmp(track->mimeType, "audio/VORBIS") == 0) {
        u_int8_t const firstByte = p[0];

        if (firstByte == 1) {  // "identification" header
            if (headerSize < 30) {
                fprintf(stderr,
                        "Vorbis \"identification\" header is too short (%d "
                        "bytes)\n",
                        headerSize);
                return False;
            } else if ((p[29] & 0x1) != 1) {
                fprintf(stderr,
                        "Vorbis \"identification\" header: 'framing_flag' is "
                        "not set\n");
                return False;
            }

            p += 7;
            u_int32_t vorbis_version =
                    (p[3] << 24) | (p[2] << 16) | (p[1] << 8) | p[0];
            p += 4;
            if (vorbis_version != 0) {
                fprintf(stderr,
                        "Vorbis \"identification\" header has a bad "
                        "'vorbis_version': 0x%08x\n",
                        vorbis_version);
                return False;
            }

            u_int8_t audio_channels = *p++;
            if (audio_channels == 0) {
                fprintf(stderr,
                        "Vorbis \"identification\" header: 'audio_channels' is "
                        "0!\n");
                return False;
            }
            track->numChannels = audio_channels;

            u_int32_t audio_sample_rate =
                    (p[3] << 24) | (p[2] << 16) | (p[1] << 8) | p[0];
            p += 4;
            if (audio_sample_rate == 0) {
                fprintf(stderr,
                        "Vorbis \"identification\" header: 'audio_sample_rate' "
                        "is 0!\n");
                return False;
            }
            track->samplingFrequency = audio_sample_rate;

            p += 4;  // skip over 'bitrate_maximum'
            u_int32_t bitrate_nominal =
                    (p[3] << 24) | (p[2] << 16) | (p[1] << 8) | p[0];
            p += 4;
            if (bitrate_nominal > 0)
                track->estBitrate = (bitrate_nominal + 500) / 1000;  // round

            p += 4;  // skip over 'bitrate_maximum'

            // Note the two 'block sizes' (samples per packet), and their
            // durations in microseconds:
            u_int8_t blocksizeBits = *p++;
            unsigned& blocksize_0 = track->vtoHdrs.blocksize[0];  // alias
            unsigned& blocksize_1 = track->vtoHdrs.blocksize[1];  // alias
            blocksize_0 = blocksizeFromExponent(blocksizeBits & 0x0F);
            blocksize_1 = blocksizeFromExponent(blocksizeBits >> 4);

            double uSecsPerSample = 1000000.0 / (track->samplingFrequency * 2);
            // Why the "2"?  I don't know, but it seems to be necessary
            track->vtoHdrs.uSecsPerPacket[0] =
                    (unsigned)(uSecsPerSample * blocksize_0);
            track->vtoHdrs.uSecsPerPacket[1] =
                    (unsigned)(uSecsPerSample * blocksize_1);
#ifdef DEBUG
            fprintf(stderr,
                    "\t%u Hz, %u-channel, %u kbps (est), block sizes: %u,%u "
                    "(%u,%u us)\n",
                    track->samplingFrequency, track->numChannels,
                    track->estBitrate, blocksize_0, blocksize_1,
                    track->vtoHdrs.uSecsPerPacket[0],
                    track->vtoHdrs.uSecsPerPacket[1]);
#endif
            // To be valid, "blocksize_0" must be <= "blocksize_1", and both
            // must be in [64,8192]:
            if (!(blocksize_0 <= blocksize_1 && blocksize_0 >= 64 &&
                  blocksize_1 <= 8192)) {
                fprintf(stderr,
                        "Invalid Vorbis \"blocksize_0\" (%d) and/or "
                        "\"blocksize_1\" (%d)!\n",
                        blocksize_0, blocksize_1);
                return False;
            }
        } else if (firstByte == 3) {  // "comment" header
            if (!validateCommentHeader(p, headerSize)) return False;
        } else if (firstByte == 5) {  // "setup" header
            // Parse the "setup" header to get the values that we want:
            // "vorbis_mode_count", and "vorbis_mode_blockflag" for each mode.
            // Unfortunately these come near the end of the header, so we have
            // to parse lots of other crap first.
            p += 7;
            if (!parseVorbisSetupHeader(track, p, headerSize)) {
                fprintf(stderr, "Failed to parse Vorbis \"setup\" header!\n");
                return False;
            }
        }
    } else if (strcmp(track->mimeType, "video/THEORA") == 0) {
        u_int8_t const firstByte = p[0];

        if (firstByte == 0x80) {  // "identification" header
            if (headerSize < 42) {
                fprintf(stderr,
                        "Theora \"identification\" header is too short (%d "
                        "bytes)\n",
                        headerSize);
                return False;
            } else if ((p[41] & 0x7) != 0) {
                fprintf(stderr,
                        "Theora \"identification\" header: 'res' bits are "
                        "non-zero\n");
                return False;
            }

            track->vtoHdrs.KFGSHIFT = ((p[40] & 3) << 3) | (p[41] >> 5);
            u_int32_t FRN = (p[22] << 24) | (p[23] << 16) | (p[24] << 8) |
                            p[25];  // Frame rate numerator
            u_int32_t FRD = (p[26] << 24) | (p[27] << 16) | (p[28] << 8) |
                            p[29];  // Frame rate numerator
#ifdef DEBUG
            fprintf(stderr,
                    "\tKFGSHIFT %d, Frame rate numerator %d, Frame rate "
                    "denominator %d\n",
                    track->vtoHdrs.KFGSHIFT, FRN, FRD);
#endif
            if (FRN == 0 || FRD == 0) {
                fprintf(stderr,
                        "Theora \"identification\" header: Bad FRN and/or FRD "
                        "values: %d, %d\n",
                        FRN, FRD);
                return False;
            }
            track->vtoHdrs.uSecsPerFrame = (unsigned)((1000000.0 * FRD) / FRN);
#ifdef DEBUG
            fprintf(stderr, "\t\t=> %u microseconds per frame\n",
                    track->vtoHdrs.uSecsPerFrame);
#endif
        } else if (firstByte == 0x81) {  // "comment" header
            if (!validateCommentHeader(p, headerSize)) return False;
        } else if (firstByte == 0x82) {  // "setup" header
            // We don't care about the contents of the Theora "setup" header;
            // just assume it's valid
        }
    } else {  // Opus audio
        if (strncmp((char const*)p, "OpusHead", 8) ==
            0) {  // "identification" header
            // Just check the size, and the 'major' number of the version byte:
            if (headerSize < 19 || (p[8] & 0xF0) != 0) return False;
        } else {  // comment header
            if (!validateCommentHeader(p, headerSize, 1 /*isOpus*/))
                return False;
        }
    }

    return True;
}

void OggFileParser::parseAndDeliverPages() {
#ifdef DEBUG
    fprintf(stderr, "parsing and delivering data\n");
#endif
    while (parseAndDeliverPage()) {
    }
}

Boolean OggFileParser::parseAndDeliverPage() {
    u_int8_t header_type_flag;
    u_int32_t bitstream_serial_number;
    parseStartOfPage(header_type_flag, bitstream_serial_number);

    OggDemuxedTrack* demuxedTrack =
            fOurDemux->lookupDemuxedTrack(bitstream_serial_number);
    if (demuxedTrack == NULL) {  // this track is not being read
#ifdef DEBUG
        fprintf(stderr,
                "\tIgnoring page from unread track; skipping %d remaining "
                "packet data bytes\n",
                fPacketSizeTable->totSizes);
#endif
        skipBytes(fPacketSizeTable->totSizes);
        return True;
    } else if (fPacketSizeTable->totSizes == 0) {
        // This page is empty (has no packets).  Skip it and continue
#ifdef DEBUG
        fprintf(stderr, "\t[track: %s] Skipping empty page\n",
                demuxedTrack->MIMEtype());
#endif
        return True;
    }

    // Start delivering packets next:
    demuxedTrack->fCurrentPageIsContinuation = (header_type_flag & 0x01) != 0;
    fCurrentTrackNumber = bitstream_serial_number;
    fCurrentParseState = DELIVERING_PACKET_WITHIN_PAGE;
    saveParserState();
    return False;
}

Boolean OggFileParser::deliverPacketWithinPage() {
    OggDemuxedTrack* demuxedTrack =
            fOurDemux->lookupDemuxedTrack(fCurrentTrackNumber);
    if (demuxedTrack == NULL) return False;  // should not happen

    unsigned packetNum = fPacketSizeTable->nextPacketNumToDeliver;
    unsigned packetSize = fPacketSizeTable->size[packetNum];

    if (!demuxedTrack->isCurrentlyAwaitingData()) {
        // Someone has been reading this stream, but isn't right now.
        // We can't deliver this frame until he asks for it, so punt for now.
        // The next time he asks for a frame, he'll get it.
#ifdef DEBUG
        fprintf(stderr,
                "\t[track: %s] Deferring delivery of packet %d (%d bytes%s)\n",
                demuxedTrack->MIMEtype(), packetNum, packetSize,
                packetNum == fPacketSizeTable->numCompletedPackets
                        ? " (incomplete)"
                        : "");
#endif
        return True;
    }

    // Deliver the next packet:
#ifdef DEBUG
    fprintf(stderr, "\t[track: %s] Delivering packet %d (%d bytes%s)\n",
            demuxedTrack->MIMEtype(), packetNum, packetSize,
            packetNum == fPacketSizeTable->numCompletedPackets ? " (incomplete)"
                                                               : "");
#endif
    unsigned numBytesDelivered = packetSize < demuxedTrack->maxSize()
                                         ? packetSize
                                         : demuxedTrack->maxSize();
    getBytes(demuxedTrack->to(), numBytesDelivered);
    u_int8_t firstByte = numBytesDelivered > 0 ? demuxedTrack->to()[0] : 0x00;
    u_int8_t secondByte = numBytesDelivered > 1 ? demuxedTrack->to()[1] : 0x00;
    demuxedTrack->to() += numBytesDelivered;

    if (demuxedTrack->fCurrentPageIsContinuation) {  // the previous page's read
                                                     // was incomplete
        demuxedTrack->frameSize() += numBytesDelivered;
    } else {
        // This is the first delivery for this "doGetNextFrame()" call.
        demuxedTrack->frameSize() = numBytesDelivered;
    }
    if (packetSize > demuxedTrack->maxSize()) {
        demuxedTrack->numTruncatedBytes() +=
                packetSize - demuxedTrack->maxSize();
    }
    demuxedTrack->maxSize() -= numBytesDelivered;

    // Figure out the duration and presentation time of this frame.
    unsigned durationInMicroseconds;
    OggTrack* track = fOurFile.lookup(demuxedTrack->fOurTrackNumber);

    if (strcmp(track->mimeType, "audio/VORBIS") == 0) {
        if ((firstByte & 0x01) != 0) {  // This is a header packet
            durationInMicroseconds = 0;
        } else {  // This is a data packet.
            // Parse the first byte to figure out its duration.
            // Extract the next "track->vtoHdrs.ilog_vorbis_mode_count_minus_1"
            // bits of the first byte:
            u_int8_t const mask =
                    0xFE << (track->vtoHdrs.ilog_vorbis_mode_count_minus_1);
            u_int8_t const modeNumber = (firstByte & ~mask) >> 1;
            if (modeNumber >= track->vtoHdrs.vorbis_mode_count) {
                fprintf(stderr,
                        "Error: Bad mode number %d (>= vorbis_mode_count %d) "
                        "in Vorbis packet!\n",
                        modeNumber, track->vtoHdrs.vorbis_mode_count);
                durationInMicroseconds = 0;
            } else {
                unsigned blockNumber =
                        track->vtoHdrs.vorbis_mode_blockflag[modeNumber];
                durationInMicroseconds =
                        track->vtoHdrs.uSecsPerPacket[blockNumber];
            }
        }
    } else if (strcmp(track->mimeType, "video/THEORA") == 0) {
        if ((firstByte & 0x80) != 0) {  // This is a header packet
            durationInMicroseconds = 0;
        } else {  // This is a data packet.
            durationInMicroseconds = track->vtoHdrs.uSecsPerFrame;
        }
    } else {  // "audio/OPUS"
        if (firstByte == 0x4F /*'O'*/ &&
            secondByte == 0x70 /*'p*/) {  // This is a header packet
            durationInMicroseconds = 0;
        } else {  // This is a data packet.
            // Parse the first byte to figure out the duration of each frame,
            // and then (if necessary) parse the second byte to figure out how
            // many frames are in this packet:
            u_int8_t config = firstByte >> 3;
            u_int8_t c = firstByte & 0x03;
            unsigned const configDuration[32] = {
                    // in microseconds
                    10000, 20000, 40000, 60000,  // config 0..3
                    10000, 20000, 40000, 60000,  // config 4..7
                    10000, 20000, 40000, 60000,  // config 8..11
                    10000, 20000,                // config 12..13
                    10000, 20000,                // config 14..15
                    2500,  5000,  10000, 20000,  // config 16..19
                    2500,  5000,  10000, 20000,  // config 20..23
                    2500,  5000,  10000, 20000,  // config 24..27
                    2500,  5000,  10000, 20000   // config 28..31
            };
            unsigned const numFramesInPacket =
                    c == 0 ? 1 : c == 3 ? (secondByte & 0x3F) : 2;
            durationInMicroseconds = numFramesInPacket * configDuration[config];
        }
    }

    if (demuxedTrack->nextPresentationTime().tv_sec == 0 &&
        demuxedTrack->nextPresentationTime().tv_usec == 0) {
        // This is the first delivery.  Initialize
        // "demuxedTrack->nextPresentationTime()":
        gettimeofday(&demuxedTrack->nextPresentationTime(), NULL);
    }
    demuxedTrack->presentationTime() = demuxedTrack->nextPresentationTime();
    demuxedTrack->durationInMicroseconds() = durationInMicroseconds;

    demuxedTrack->nextPresentationTime().tv_usec += durationInMicroseconds;
    while (demuxedTrack->nextPresentationTime().tv_usec >= 1000000) {
        ++demuxedTrack->nextPresentationTime().tv_sec;
        demuxedTrack->nextPresentationTime().tv_usec -= 1000000;
    }
    saveParserState();

    // And check whether there's a next packet in this page:
    if (packetNum == fPacketSizeTable->numCompletedPackets) {
        // This delivery was for an incomplete packet, at the end of the page.
        // Return without completing delivery:
        fCurrentParseState = PARSING_AND_DELIVERING_PAGES;
        return False;
    }

    if (packetNum < fPacketSizeTable->numCompletedPackets - 1 ||
        fPacketSizeTable->lastPacketIsIncomplete) {
        // There is at least one more packet (possibly incomplete) left in this
        // packet. Deliver it next:
        ++fPacketSizeTable->nextPacketNumToDeliver;
    } else {
        // Start parsing a new page next:
        fCurrentParseState = PARSING_AND_DELIVERING_PAGES;
    }

    FramedSource::afterGetting(demuxedTrack);  // completes delivery
    return True;
}

void OggFileParser::parseStartOfPage(u_int8_t& header_type_flag,
                                     u_int32_t& bitstream_serial_number) {
    saveParserState();
    // First, make sure we start with the 'capture_pattern': 0x4F676753
    // ('OggS'):
    while (test4Bytes() != 0x4F676753) {
        skipBytes(1);
        saveParserState();  // ensures forward progress through the file
    }
    skipBytes(4);
#ifdef DEBUG
    fprintf(stderr, "\nSaw Ogg page header:\n");
#endif

    u_int8_t stream_structure_version = get1Byte();
    if (stream_structure_version != 0) {
        fprintf(stderr,
                "Saw page with unknown Ogg file version number: 0x%02x\n",
                stream_structure_version);
    }

    header_type_flag = get1Byte();
#ifdef DEBUG
    fprintf(stderr, "\theader_type_flag: 0x%02x (", header_type_flag);
    if (header_type_flag & 0x01) fprintf(stderr, "continuation ");
    if (header_type_flag & 0x02) fprintf(stderr, "bos ");
    if (header_type_flag & 0x04) fprintf(stderr, "eos ");
    fprintf(stderr, ")\n");
#endif

    u_int32_t granule_position1 = byteSwap(get4Bytes());
    u_int32_t granule_position2 = byteSwap(get4Bytes());
    bitstream_serial_number = byteSwap(get4Bytes());
    u_int32_t page_sequence_number = byteSwap(get4Bytes());
    u_int32_t CRC_checksum = byteSwap(get4Bytes());
    u_int8_t number_page_segments = get1Byte();
#ifdef DEBUG
    fprintf(stderr,
            "\tgranule_position 0x%08x%08x, bitstream_serial_number 0x%08x, "
            "page_sequence_number 0x%08x, CRC_checksum 0x%08x, "
            "number_page_segments %d\n",
            granule_position2, granule_position1, bitstream_serial_number,
            page_sequence_number, CRC_checksum, number_page_segments);
#else
    // Dummy statements to prevent 'unused variable' compiler warnings:
#define DUMMY_STATEMENT(x) \
    do {                   \
        x = x;             \
    } while (0)
    DUMMY_STATEMENT(granule_position1);
    DUMMY_STATEMENT(granule_position2);
    DUMMY_STATEMENT(page_sequence_number);
    DUMMY_STATEMENT(CRC_checksum);
#endif

    // Look at the "segment_table" to count the sizes of the packets in this
    // page:
    delete fPacketSizeTable /*if any*/;
    fPacketSizeTable = new PacketSizeTable(number_page_segments);
    u_int8_t lacing_value = 0;
#ifdef DEBUG
    fprintf(stderr, "\tsegment_table\n");
#endif
    for (unsigned i = 0; i < number_page_segments; ++i) {
        lacing_value = get1Byte();
#ifdef DEBUG
        fprintf(stderr, "\t\t%d:\t%d", i, lacing_value);
#endif
        fPacketSizeTable->totSizes += lacing_value;
        fPacketSizeTable->size[fPacketSizeTable->numCompletedPackets] +=
                lacing_value;
        if (lacing_value < 255) {
            // This completes a packet:
#ifdef DEBUG
            fprintf(stderr, " (->%d)",
                    fPacketSizeTable
                            ->size[fPacketSizeTable->numCompletedPackets]);
#endif
            ++fPacketSizeTable->numCompletedPackets;
        }
#ifdef DEBUG
        fprintf(stderr, "\n");
#endif
    }

    fPacketSizeTable->lastPacketIsIncomplete = lacing_value == 255;
}
