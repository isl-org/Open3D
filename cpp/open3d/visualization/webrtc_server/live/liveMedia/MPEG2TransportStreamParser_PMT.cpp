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
// A parser for a MPEG Transport Stream
// Implementation

#include "MPEG2TransportStreamParser.hh"

void MPEG2TransportStreamParser ::parsePMT(PIDState_PMT* pidState,
                                           Boolean pusi,
                                           unsigned numDataBytes) {
#ifdef DEBUG_CONTENTS
    fprintf(stderr, "\tProgram Map Table\n");
#endif
    unsigned startPos = curOffset();

    do {
        if (pusi) {
            u_int8_t pointer_field = get1Byte();
            skipBytes(pointer_field);  // usually 0
        }

        u_int8_t table_id = get1Byte();
        if (table_id != 0x02) {
#ifdef DEBUG_ERRORS
            fprintf(stderr,
                    "MPEG2TransportStreamParser::parsePMT(0x%04x, %d, %d): bad "
                    "table_id: 0x%02x\n",
                    pidState->PID, pusi, numDataBytes, table_id);
#endif
            break;
        }

        u_int16_t flagsPlusSection_length = get2Bytes();
        u_int16_t section_length = flagsPlusSection_length & 0x0FFF;
#ifdef DEBUG_CONTENTS
        fprintf(stderr, "\t\tsection_length: %d\n", section_length);
#endif
        if (section_length < 13 /*too small for remaining fields + CRC*/ ||
            section_length > 1021 /*as per specification*/) {
#ifdef DEBUG_ERRORS
            fprintf(stderr,
                    "MPEG2TransportStreamParser::parsePMT(0x%04x, %d, %d): Bad "
                    "section_length: %d\n",
                    pidState->PID, pusi, numDataBytes, section_length);
#endif
            break;
        }
        unsigned endPos = curOffset() + section_length;
        if (endPos - startPos > numDataBytes) {
#ifdef DEBUG_ERRORS
            fprintf(stderr,
                    "MPEG2TransportStreamParser::parsePMT(0x%04x, %d, %d): "
                    "section_length %d gives us a total size %d that's too "
                    "large!\n",
                    pidState->PID, pusi, numDataBytes, section_length,
                    endPos - startPos);
#endif
            break;
        }

        u_int16_t program_number = get2Bytes();
        if (program_number != pidState->program_number) {
#ifdef DEBUG_ERRORS
            fprintf(stderr,
                    "MPEG2TransportStreamParser::parsePMT(0x%04x, %d, %d): "
                    "program_number %d does not match the value %d that was "
                    "given to us in the PAT!\n",
                    pidState->PID, pusi, numDataBytes, program_number,
                    pidState->program_number);
#endif
            break;
        }
#ifdef DEBUG_CONTENTS
        fprintf(stderr, "\t\tprogram_number: %d\n", program_number);

        u_int8_t version_number_byte = get1Byte();
        u_int8_t version_number = (version_number_byte & 0x1E) >> 1;
        u_int8_t section_number = get1Byte();
        u_int8_t last_section_number = get1Byte();
        fprintf(stderr,
                "\t\tversion_number: %d; section_number: %d; "
                "last_section_number: %d\n",
                version_number, section_number, last_section_number);
        u_int16_t PCR_PID = get2Bytes();
        PCR_PID &= 0x1FFF;
        fprintf(stderr, "\t\tPCR_PID: 0x%04x\n", PCR_PID);
#else
        skipBytes(5);
#endif

        u_int16_t program_info_length = get2Bytes();
        program_info_length &= 0x0FFF;
#ifdef DEBUG_CONTENTS
        fprintf(stderr, "\t\tprogram_info_length: %d\n", program_info_length);
#endif
        unsigned endOfDescriptors = curOffset() + program_info_length;
        if (endOfDescriptors + 4 /*CRC*/ - startPos > numDataBytes) {
#ifdef DEBUG_ERRORS
            fprintf(stderr,
                    "MPEG2TransportStreamParser::parsePMT(0x%04x, %d, %d): "
                    "program_info_length %d gives us a total size %d that's "
                    "too large!\n",
                    pidState->PID, pusi, numDataBytes, program_info_length,
                    endOfDescriptors + 4 - startPos);
#endif
            break;
        }
        parseStreamDescriptors(program_info_length);

        while (curOffset() <=
               endPos - 4 /*for CRC*/ - 5 /*for mapping fields*/) {
            u_int8_t stream_type = get1Byte();
            u_int16_t elementary_PID = get2Bytes();
            elementary_PID &= 0x1FFF;
            u_int16_t ES_info_length = get2Bytes();
            ES_info_length &= 0x0FFF;
#ifdef DEBUG_CONTENTS
            extern StreamType StreamTypes[];
            char const* const streamTypeDesc =
                    StreamTypes[stream_type].description;
            fprintf(stderr,
                    "\t\tstream_type: 0x%02x (%s); elementary_PID: 0x%04x; "
                    "ES_info_length: %d\n",
                    stream_type,
                    streamTypeDesc == NULL ? "???" : streamTypeDesc,
                    elementary_PID, ES_info_length);
#endif
            endOfDescriptors = curOffset() + ES_info_length;
            if (endOfDescriptors + 4 /*CRC*/ - startPos > numDataBytes) {
#ifdef DEBUG_ERRORS
                fprintf(stderr,
                        "MPEG2TransportStreamParser::parsePMT(0x%04x, %d, %d): "
                        "ES_info_length %d gives us a total size %d that's too "
                        "large!\n",
                        pidState->PID, pusi, numDataBytes, ES_info_length,
                        endOfDescriptors + 4 - startPos);
#endif
                break;
            }
            parseStreamDescriptors(ES_info_length);

            if (fPIDState[elementary_PID] == NULL) {
                fPIDState[elementary_PID] = new PIDState_STREAM(
                        *this, elementary_PID, program_number, stream_type);
            }
        }
    } while (0);

    // Skip (ignore) all remaining bytes in this packet (including the CRC):
    int numBytesLeft = numDataBytes - (curOffset() - startPos);
    if (numBytesLeft > 0) {
#ifdef DEBUG_CONTENTS
        fprintf(stderr, "\t\t+%d CRC and stuffing bytes\n", numBytesLeft);
#endif
        skipBytes(numBytesLeft);
    }
}

#ifdef DEBUG_CONTENTS
#define pDesc(str)                                                            \
    do {                                                                      \
        fprintf(stderr,                                                       \
                "\t\t\tdescriptor_tag: 0x%02x (%s); descriptor_length: %d\n", \
                descriptor_tag, (str), descriptor_length);                    \
    } while (0)
#else
#define pDesc(str)
#endif

void MPEG2TransportStreamParser::parseStreamDescriptors(
        unsigned numDescriptorBytes) {
    while (numDescriptorBytes >=
           2 /* enough for "descriptor_tag" and "descriptor_length" */) {
        u_int8_t descriptor_tag = get1Byte();
        u_int8_t descriptor_length = get1Byte();
        numDescriptorBytes -= 2;

        if (descriptor_length > numDescriptorBytes) {
#ifdef DEBUG_ERRORS
            fprintf(stderr,
                    "MPEG2TransportStreamParser::parseStreamDescriptors() "
                    "error: Saw descriptor_length %d > remaining bytes %d\n",
                    descriptor_length, numDescriptorBytes);
#endif
            skipBytes(numDescriptorBytes);
            numDescriptorBytes = 0;
            break;
        }

        Boolean parsedDescriptor = False;
        switch (descriptor_tag) {
                // Note: These are the tags that we've seen to date.  Add more
                // when we see more.
            case 0x02: {
                pDesc("video");
                if (descriptor_length < 1) break;
                u_int8_t flags = get1Byte();
                Boolean MPEG_1_only_flag = (flags & 0x04) != 0;
#ifdef DEBUG_CONTENTS
                fprintf(stderr,
                        "\t\t\t\tflags: 0x%02x (frame_rate_code 0x%1x; "
                        "MPEG_1_only_flag %d)\n",
                        flags, (flags & 0x78) >> 3, MPEG_1_only_flag);
#endif
                if (MPEG_1_only_flag == 0) {
                    if (descriptor_length < 3) break;
#ifdef DEBUG_CONTENTS
                    u_int8_t profile_and_level_indication = get1Byte();
                    flags = get1Byte();
                    fprintf(stderr,
                            "\t\t\t\tprofile_and_level_indication 0x%02x; "
                            "flags 0x%02x (chroma_format 0x%1x)\n",
                            profile_and_level_indication, flags,
                            (flags & 0xC0) >> 6);
#else
                    skipBytes(2);
#endif
                }
                numDescriptorBytes -= descriptor_length;
                parsedDescriptor = True;
                break;
            }
            case 0x03: {
                pDesc("audio");
                if (descriptor_length < 1) break;
#ifdef DEBUG_CONTENTS
                u_int8_t flags = get1Byte();
                fprintf(stderr, "\t\t\t\tflags: 0x%02x (layer %d)\n", flags,
                        (flags & 0x30) >> 4);
#else
                skipBytes(1);
#endif
                numDescriptorBytes -= descriptor_length;
                parsedDescriptor = True;
                break;
            }
            case 0x05: {
                pDesc("registration");
                if (descriptor_length < 4) break;
#ifdef DEBUG_CONTENTS
                u_int32_t format_identifier = get4Bytes();
                fprintf(stderr,
                        "\t\t\t\tformat_identifier: 0x%08x (%c%c%c%c)\n",
                        format_identifier, format_identifier >> 24,
                        format_identifier >> 16, format_identifier >> 8,
                        format_identifier);
                if (descriptor_length > 4) {
                    fprintf(stderr, "\t\t\t\tadditional_identification_info: ");
                    for (unsigned i = 4; i < descriptor_length; ++i)
                        fprintf(stderr, "%02x:", get1Byte());
                    fprintf(stderr, "\n");
                }
#else
                skipBytes(descriptor_length);
#endif
                numDescriptorBytes -= descriptor_length;
                parsedDescriptor = True;
                break;
            }
            case 0x06: {
                pDesc("data stream alignment");
                if (descriptor_length < 1) break;
#ifdef DEBUG_CONTENTS
                u_int8_t alignment_type = get1Byte();
                fprintf(stderr, "\t\t\t\talignment_type: 0x%02x\n",
                        alignment_type);
#else
                skipBytes(1);
#endif
                numDescriptorBytes -= descriptor_length;
                parsedDescriptor = True;
                break;
            }
            case 0x0a: {
                pDesc("ISO 639 language descriptor");
                for (unsigned i = 0; i < descriptor_length / 4; ++i) {
#ifdef DEBUG_CONTENTS
                    fprintf(stderr,
                            "\t\t\t\tISO_639_language_code: %c%c%c; "
                            "audio_type: 0x%02x\n",
                            get1Byte(), get1Byte(), get1Byte(), get1Byte());
#else
                    skipBytes(4);
#endif
                }
                numDescriptorBytes -= descriptor_length;
                parsedDescriptor = True;
                break;
            }
            case 0x0b: {
                pDesc("system clock");
                if (descriptor_length < 2) break;
#ifdef DEBUG_CONTENTS
                u_int8_t flags = get1Byte();
                Boolean external_clock_ref = (flags & 0x80) != 0;
                u_int8_t clock_accuracy_integer = flags & 0x3F;

                u_int8_t clock_accuracy_exponent = get1Byte();
                clock_accuracy_exponent >>= 5;
                float ppm = clock_accuracy_integer * 1.0;
                for (unsigned i = 0; i < clock_accuracy_exponent; ++i)
                    ppm /= 10.0;
                fprintf(stderr,
                        "\t\t\t\texternal_clock: %d; clock_accuracy int: %d, "
                        "exp: %d -> %f ppm\n",
                        external_clock_ref, clock_accuracy_integer,
                        clock_accuracy_exponent, ppm);
#else
                skipBytes(2);
#endif
                numDescriptorBytes -= descriptor_length;
                parsedDescriptor = True;
                break;
            }
            case 0x0e: {
                pDesc("maximum bitrate");
                if (descriptor_length < 3) break;
#ifdef DEBUG_CONTENTS
                u_int32_t maximum_bitrate =
                        ((get1Byte() & 0x3F) << 16) | get2Bytes();  // 22 bits
                fprintf(stderr, "\t\t\t\tmaximum_bitrate: %d => %f Mbps\n",
                        maximum_bitrate,
                        (maximum_bitrate * 50 * 8) / 1000000.0);
#else
                skipBytes(3);
#endif
                numDescriptorBytes -= descriptor_length;
                parsedDescriptor = True;
                break;
            }
            case 0x10: {
                pDesc("smoothing buffer");
                if (descriptor_length < 6) break;
#ifdef DEBUG_CONTENTS
                u_int32_t sb_leak_rate =
                        ((get1Byte() & 0x3F) << 16) | get2Bytes();  // 22 bits
                u_int32_t sb_size =
                        ((get1Byte() & 0x3F) << 16) | get2Bytes();  // 22 bits
                fprintf(stderr,
                        "\t\t\t\tsb_leak_rate: %d => %f Mbps; sb_size: %d "
                        "bytes\n",
                        sb_leak_rate, (sb_leak_rate * 400) / 1000000.0,
                        sb_size);
#else
                skipBytes(6);
#endif
                numDescriptorBytes -= descriptor_length;
                parsedDescriptor = True;
                break;
            }
            case 0x1d: {
                pDesc("IOD parameters for ISO/IEC 14496-1");
                // Note: We don't know how to parse this.  (Where's a document
                // that describes this?)
                skipBytes(descriptor_length);
                numDescriptorBytes -= descriptor_length;
                parsedDescriptor = True;
                break;
            }
            case 0x28: {
                pDesc("H.264 video parameters");
                if (descriptor_length < 4) break;
#ifdef DEBUG_CONTENTS
                u_int8_t profile_idc = get1Byte();
                u_int8_t flags1 = get1Byte();
                u_int8_t level_idc = get1Byte();
                u_int8_t flags2 = get1Byte();
                fprintf(stderr,
                        "\t\t\t\tprofile_idc: 0x%02x, flags1: 0x%02x, "
                        "level_idc: 0x%02x, flags2: 0x%02x\n",
                        profile_idc, flags1, level_idc, flags2);
#else
                skipBytes(4);
#endif
                numDescriptorBytes -= descriptor_length;
                parsedDescriptor = True;
                break;
            }
            case 0x52: {
                pDesc("stream identifier");
                if (descriptor_length < 1) break;
#ifdef DEBUG_CONTENTS
                u_int8_t component_tag = get1Byte();
                fprintf(stderr, "\t\t\t\tcomponent_tag: %d\n", component_tag);
#else
                skipBytes(1);
#endif
                numDescriptorBytes -= descriptor_length;
                parsedDescriptor = True;
                break;
            }
            case 0x56: {
                pDesc("teletext");
                for (unsigned i = 0; i < descriptor_length / 5; ++i) {
#ifdef DEBUG_CONTENTS
                    fprintf(stderr, "\t\t\t\tISO_639_language_code: %c%c%c",
                            get1Byte(), get1Byte(), get1Byte());
                    u_int8_t typePlusMagazine = get1Byte();
                    fprintf(stderr, "; type: 0x%02x; magazine: %d; page: %d\n",
                            typePlusMagazine >> 3, typePlusMagazine & 0x07,
                            get1Byte());
#else
                    skipBytes(5);
#endif
                }
                numDescriptorBytes -= descriptor_length;
                parsedDescriptor = True;
                break;
            }
            case 0x59: {
                pDesc("subtitling");
                for (unsigned i = 0; i < descriptor_length / 8; ++i) {
#ifdef DEBUG_CONTENTS
                    fprintf(stderr, "\t\t\t\tISO_639_language_code: %c%c%c",
                            get1Byte(), get1Byte(), get1Byte());
                    fprintf(stderr,
                            "; subtitling_type: 0x%02x; composition_page_id: "
                            "0x%04x; ancillary_page_id: 0x%04x\n",
                            get1Byte(), get2Bytes(), get2Bytes());
#else
                    skipBytes(8);
#endif
                }
                numDescriptorBytes -= descriptor_length;
                parsedDescriptor = True;
                break;
            }
            case 0x6f: {
                pDesc("application signalling");
                for (unsigned i = 0; i < descriptor_length / 3; ++i) {
#ifdef DEBUG_CONTENTS
                    fprintf(stderr,
                            "\t\t\t\tapplication_type: 0x%04x; "
                            "AIT_version_number: %d\n",
                            get2Bytes() & 0x7FFF, get1Byte() & 0x1F);
#else
                    skipBytes(3);
#endif
                }
                numDescriptorBytes -= descriptor_length;
                parsedDescriptor = True;
                break;
            }
            case 0x7a: {
                pDesc("enhanced AC-3");
                if (descriptor_length < 1) break;
#ifdef DEBUG_CONTENTS
                u_int8_t flags = get1Byte();
                fprintf(stderr, "\t\t\t\tflags: 0x%02x", flags);
                if (descriptor_length > 1) {
                    fprintf(stderr, "; extra bytes: ");
                    for (unsigned i = 1; i < descriptor_length; ++i)
                        fprintf(stderr, "0x%02x ", get1Byte());
                }
                fprintf(stderr, "\n");
#else
                skipBytes(descriptor_length);
#endif
                numDescriptorBytes -= descriptor_length;
                parsedDescriptor = True;
                break;
            }
            case 0x81: {
                pDesc("AC-3 audio");
                if (descriptor_length < 3) break;
#ifdef DEBUG_CONTENTS
                u_int8_t flags = get1Byte();
                fprintf(stderr, "\t\t\t\tsample_rate_code: %d; bsid: 0x%02x",
                        flags >> 5, flags & 0x1F);
                flags = get1Byte();
                fprintf(stderr, "; bit_rate_code: %d; surround_mode: %d",
                        flags >> 2, flags & 0x03);
                flags = get1Byte();
                fprintf(stderr, "; bsmod: %d; num_channels: %d; full_svc: %d",
                        flags >> 5, (flags & 0x1E) >> 1, (flags & 0x01));
                if (descriptor_length > 3) {
                    fprintf(stderr, "; extra bytes: ");
                    for (unsigned i = 3; i < descriptor_length; ++i)
                        fprintf(stderr, "0x%02x ", get1Byte());
                }
                fprintf(stderr, "\n");
#else
                skipBytes(descriptor_length);
#endif
                numDescriptorBytes -= descriptor_length;
                parsedDescriptor = True;
                break;
            }
            case 0x86: {
                pDesc("caption service");
                if (descriptor_length < 1) break;
                u_int8_t number_of_services = get1Byte() & 0x1F;
#ifdef DEBUG_CONTENTS
                fprintf(stderr, "\t\t\t\tnumber_of_services: %d\n",
                        number_of_services);
#endif
                if (descriptor_length < number_of_services * 6) break;
#ifdef DEBUG_CONTENTS
                for (unsigned i = 0; i < number_of_services; ++i) {
                    fprintf(stderr, "\t\t\t\t\tlanguage: %c%c%c", get1Byte(),
                            get1Byte(), get1Byte());

                    u_int8_t flags = get1Byte();
                    Boolean digital_cc = (flags & 0x80) != 0;
                    fprintf(stderr, "; digital_cc %d", digital_cc);
                    if (digital_cc == 0) {
                        fprintf(stderr, "; line21_field: %d", flags & 0x01);
                    } else {
                        fprintf(stderr, "; caption_service_number: %d",
                                flags & 0x3F);
                    }

                    u_int16_t flags2 = get2Bytes();
                    fprintf(stderr,
                            "; easy_reader: %d; wide_aspect_ratio: %d\n",
                            (flags2 & 0x8000) != 0, (flags2 & 0x4000) != 0);
                }
#else
                skipBytes(number_of_services * 6);
#endif
                numDescriptorBytes -= descriptor_length;
                parsedDescriptor = True;
                break;
            }
            default: {
                pDesc("???");
                skipBytes(descriptor_length);
                numDescriptorBytes -= descriptor_length;
                parsedDescriptor = True;
                break;
            }
        }
        if (!parsedDescriptor) break;  // an error occurred
    }

    // Skip over any remaining descriptor bytes (as a result of a parsing
    // error):
    if (numDescriptorBytes > 0) {
#ifdef DEBUG_ERRORS
        fprintf(stderr,
                "MPEG2TransportStreamParser::parseStreamDescriptors() Parsing "
                "error left %d bytes unparsed\n",
                numDescriptorBytes);
#endif
        skipBytes(numDescriptorBytes);
    }
}

//########## PIDState_PMT implementation ##########

PIDState_PMT ::PIDState_PMT(MPEG2TransportStreamParser& parser,
                            u_int16_t pid,
                            u_int16_t programNumber)
    : PIDState(parser, pid, PMT), program_number(programNumber) {}

PIDState_PMT::~PIDState_PMT() {}
