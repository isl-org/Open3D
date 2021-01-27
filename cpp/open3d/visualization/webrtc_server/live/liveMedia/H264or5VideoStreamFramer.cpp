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
// A filter that breaks up a H.264 or H.265 Video Elementary Stream into NAL
// units. Implementation

#include "H264or5VideoStreamFramer.hh"

#include "BitVector.hh"
#include "MPEGVideoStreamParser.hh"

////////// H264or5VideoStreamParser definition //////////

class H264or5VideoStreamParser : public MPEGVideoStreamParser {
public:
    H264or5VideoStreamParser(int hNumber,
                             H264or5VideoStreamFramer* usingSource,
                             FramedSource* inputSource,
                             Boolean includeStartCodeInOutput);
    virtual ~H264or5VideoStreamParser();

private:  // redefined virtual functions:
    virtual void flushInput();
    virtual unsigned parse();

private:
    H264or5VideoStreamFramer* usingSource() {
        return (H264or5VideoStreamFramer*)fUsingSource;
    }

    Boolean isVPS(u_int8_t nal_unit_type) {
        return usingSource()->isVPS(nal_unit_type);
    }
    Boolean isSPS(u_int8_t nal_unit_type) {
        return usingSource()->isSPS(nal_unit_type);
    }
    Boolean isPPS(u_int8_t nal_unit_type) {
        return usingSource()->isPPS(nal_unit_type);
    }
    Boolean isVCL(u_int8_t nal_unit_type) {
        return usingSource()->isVCL(nal_unit_type);
    }
    Boolean isSEI(u_int8_t nal_unit_type);
    Boolean isEOF(u_int8_t nal_unit_type);
    Boolean usuallyBeginsAccessUnit(u_int8_t nal_unit_type);

    void removeEmulationBytes(u_int8_t* nalUnitCopy,
                              unsigned maxSize,
                              unsigned& nalUnitCopySize);

    void analyze_video_parameter_set_data(unsigned& num_units_in_tick,
                                          unsigned& time_scale);
    void analyze_seq_parameter_set_data(unsigned& num_units_in_tick,
                                        unsigned& time_scale);
    void profile_tier_level(BitVector& bv, unsigned max_sub_layers_minus1);
    void analyze_vui_parameters(BitVector& bv,
                                unsigned& num_units_in_tick,
                                unsigned& time_scale);
    void analyze_hrd_parameters(BitVector& bv);
    void analyze_sei_data(u_int8_t nal_unit_type);
    void analyze_sei_payload(unsigned payloadType,
                             unsigned payloadSize,
                             u_int8_t* payload);

private:
    int fHNumber;  // 264 or 265
    unsigned fOutputStartCodeSize;
    Boolean fHaveSeenFirstStartCode, fHaveSeenFirstByteOfNALUnit;
    u_int8_t fFirstByteOfNALUnit;
    double fParsedFrameRate;
    // variables set & used in the specification:
    unsigned cpb_removal_delay_length_minus1, dpb_output_delay_length_minus1;
    Boolean CpbDpbDelaysPresentFlag, pic_struct_present_flag;
    double DeltaTfiDivisor;
};

////////// H264or5VideoStreamFramer implementation //////////

H264or5VideoStreamFramer ::H264or5VideoStreamFramer(
        int hNumber,
        UsageEnvironment& env,
        FramedSource* inputSource,
        Boolean createParser,
        Boolean includeStartCodeInOutput,
        Boolean insertAccessUnitDelimiters)
    : MPEGVideoStreamFramer(env, inputSource),
      fHNumber(hNumber),
      fIncludeStartCodeInOutput(includeStartCodeInOutput),
      fInsertAccessUnitDelimiters(insertAccessUnitDelimiters),
      fLastSeenVPS(NULL),
      fLastSeenVPSSize(0),
      fLastSeenSPS(NULL),
      fLastSeenSPSSize(0),
      fLastSeenPPS(NULL),
      fLastSeenPPSSize(0) {
    fParser = createParser
                      ? new H264or5VideoStreamParser(hNumber, this, inputSource,
                                                     includeStartCodeInOutput)
                      : NULL;
    fNextPresentationTime = fPresentationTimeBase;
    fFrameRate = 25.0;  // We assume a frame rate of 25 fps, unless we learn
                        // otherwise (from parsing a VPS or SPS NAL unit)
}

H264or5VideoStreamFramer::~H264or5VideoStreamFramer() {
    delete[] fLastSeenPPS;
    delete[] fLastSeenSPS;
    delete[] fLastSeenVPS;
}

#define VPS_MAX_SIZE \
    1000  // larger than the largest possible VPS (Video Parameter Set) NAL unit

void H264or5VideoStreamFramer::saveCopyOfVPS(u_int8_t* from, unsigned size) {
    if (from == NULL) return;
    delete[] fLastSeenVPS;
    fLastSeenVPS = new u_int8_t[size];
    memmove(fLastSeenVPS, from, size);

    fLastSeenVPSSize = size;
}

#define SPS_MAX_SIZE \
    1000  // larger than the largest possible SPS (Sequence Parameter Set) NAL
          // unit

void H264or5VideoStreamFramer::saveCopyOfSPS(u_int8_t* from, unsigned size) {
    if (from == NULL) return;
    delete[] fLastSeenSPS;
    fLastSeenSPS = new u_int8_t[size];
    memmove(fLastSeenSPS, from, size);

    fLastSeenSPSSize = size;
}

void H264or5VideoStreamFramer::saveCopyOfPPS(u_int8_t* from, unsigned size) {
    if (from == NULL) return;
    delete[] fLastSeenPPS;
    fLastSeenPPS = new u_int8_t[size];
    memmove(fLastSeenPPS, from, size);

    fLastSeenPPSSize = size;
}

Boolean H264or5VideoStreamFramer::isVPS(u_int8_t nal_unit_type) {
    // VPS NAL units occur in H.265 only:
    return fHNumber == 265 && nal_unit_type == 32;
}

Boolean H264or5VideoStreamFramer::isSPS(u_int8_t nal_unit_type) {
    return fHNumber == 264 ? nal_unit_type == 7 : nal_unit_type == 33;
}

Boolean H264or5VideoStreamFramer::isPPS(u_int8_t nal_unit_type) {
    return fHNumber == 264 ? nal_unit_type == 8 : nal_unit_type == 34;
}

Boolean H264or5VideoStreamFramer::isVCL(u_int8_t nal_unit_type) {
    return fHNumber == 264 ? (nal_unit_type <= 5 && nal_unit_type > 0)
                           : (nal_unit_type <= 31);
}

void H264or5VideoStreamFramer::doGetNextFrame() {
    if (fInsertAccessUnitDelimiters && pictureEndMarker()) {
        // Deliver an "access_unit_delimiter" NAL unit instead:
        unsigned const startCodeSize = fIncludeStartCodeInOutput ? 4 : 0;
        unsigned const audNALSize = fHNumber == 264 ? 2 : 3;

        fFrameSize = startCodeSize + audNALSize;
        if (fFrameSize > fMaxSize) {  // there's no space
            fNumTruncatedBytes = fFrameSize - fMaxSize;
            fFrameSize = fMaxSize;
            handleClosure();
            return;
        }

        if (fIncludeStartCodeInOutput) {
            *fTo++ = 0x00;
            *fTo++ = 0x00;
            *fTo++ = 0x00;
            *fTo++ = 0x01;
        }
        if (fHNumber == 264) {
            *fTo++ = 9;        // "Access unit delimiter" nal_unit_type
            *fTo++ = 0xF0;     // "primary_pic_type" (7); "rbsp_trailing_bits()"
        } else {               // H.265
            *fTo++ = 35 << 1;  // "Access unit delimiter" nal_unit_type
            *fTo++ = 0;  // "nuh_layer_id" (0); "nuh_temporal_id_plus1" (0) (Is
                         // this correct??)
            *fTo++ = 0x50;  // "pic_type" (2); "rbsp_trailing_bits()" (Is this
                            // correct??)
        }

        pictureEndMarker() = False;  // for next time
        afterGetting(this);
    } else {
        // Do the normal delivery of a NAL unit from the parser:
        MPEGVideoStreamFramer::doGetNextFrame();
    }
}

////////// H264or5VideoStreamParser implementation //////////

H264or5VideoStreamParser ::H264or5VideoStreamParser(
        int hNumber,
        H264or5VideoStreamFramer* usingSource,
        FramedSource* inputSource,
        Boolean includeStartCodeInOutput)
    : MPEGVideoStreamParser(usingSource, inputSource),
      fHNumber(hNumber),
      fOutputStartCodeSize(includeStartCodeInOutput ? 4 : 0),
      fHaveSeenFirstStartCode(False),
      fHaveSeenFirstByteOfNALUnit(False),
      fParsedFrameRate(0.0),
      cpb_removal_delay_length_minus1(23),
      dpb_output_delay_length_minus1(23),
      CpbDpbDelaysPresentFlag(0),
      pic_struct_present_flag(0),
      DeltaTfiDivisor(2.0) {}

H264or5VideoStreamParser::~H264or5VideoStreamParser() {}

#define PREFIX_SEI_NUT 39  // for H.265
#define SUFFIX_SEI_NUT 40  // for H.265
Boolean H264or5VideoStreamParser::isSEI(u_int8_t nal_unit_type) {
    return fHNumber == 264 ? nal_unit_type == 6
                           : (nal_unit_type == PREFIX_SEI_NUT ||
                              nal_unit_type == SUFFIX_SEI_NUT);
}

Boolean H264or5VideoStreamParser::isEOF(u_int8_t nal_unit_type) {
    // "end of sequence" or "end of (bit)stream"
    return fHNumber == 264 ? (nal_unit_type == 10 || nal_unit_type == 11)
                           : (nal_unit_type == 36 || nal_unit_type == 37);
}

Boolean H264or5VideoStreamParser::usuallyBeginsAccessUnit(
        u_int8_t nal_unit_type) {
    return fHNumber == 264
                   ? (nal_unit_type >= 6 && nal_unit_type <= 9) ||
                             (nal_unit_type >= 14 && nal_unit_type <= 18)
                   : (nal_unit_type >= 32 && nal_unit_type <= 35) ||
                             (nal_unit_type == 39) ||
                             (nal_unit_type >= 41 && nal_unit_type <= 44) ||
                             (nal_unit_type >= 48 && nal_unit_type <= 55);
}

void H264or5VideoStreamParser ::removeEmulationBytes(
        u_int8_t* nalUnitCopy, unsigned maxSize, unsigned& nalUnitCopySize) {
    u_int8_t const* nalUnitOrig = fStartOfFrame + fOutputStartCodeSize;
    unsigned const numBytesInNALunit = fTo - nalUnitOrig;
    nalUnitCopySize = removeH264or5EmulationBytes(
            nalUnitCopy, maxSize, nalUnitOrig, numBytesInNALunit);
}

#ifdef DEBUG
char const* nal_unit_type_description_h264[32] = {
        "Unspecified",                                                     // 0
        "Coded slice of a non-IDR picture",                                // 1
        "Coded slice data partition A",                                    // 2
        "Coded slice data partition B",                                    // 3
        "Coded slice data partition C",                                    // 4
        "Coded slice of an IDR picture",                                   // 5
        "Supplemental enhancement information (SEI)",                      // 6
        "Sequence parameter set",                                          // 7
        "Picture parameter set",                                           // 8
        "Access unit delimiter",                                           // 9
        "End of sequence",                                                 // 10
        "End of stream",                                                   // 11
        "Filler data",                                                     // 12
        "Sequence parameter set extension",                                // 13
        "Prefix NAL unit",                                                 // 14
        "Subset sequence parameter set",                                   // 15
        "Reserved",                                                        // 16
        "Reserved",                                                        // 17
        "Reserved",                                                        // 18
        "Coded slice of an auxiliary coded picture without partitioning",  // 19
        "Coded slice extension",                                           // 20
        "Reserved",                                                        // 21
        "Reserved",                                                        // 22
        "Reserved",                                                        // 23
        "Unspecified",                                                     // 24
        "Unspecified",                                                     // 25
        "Unspecified",                                                     // 26
        "Unspecified",                                                     // 27
        "Unspecified",                                                     // 28
        "Unspecified",                                                     // 29
        "Unspecified",                                                     // 30
        "Unspecified"                                                      // 31
};
char const* nal_unit_type_description_h265[64] = {
        "Coded slice segment of a non-TSA, non-STSA trailing picture",  // 0
        "Coded slice segment of a non-TSA, non-STSA trailing picture",  // 1
        "Coded slice segment of a TSA picture",                         // 2
        "Coded slice segment of a TSA picture",                         // 3
        "Coded slice segment of a STSA picture",                        // 4
        "Coded slice segment of a STSA picture",                        // 5
        "Coded slice segment of a RADL picture",                        // 6
        "Coded slice segment of a RADL picture",                        // 7
        "Coded slice segment of a RASL picture",                        // 8
        "Coded slice segment of a RASL picture",                        // 9
        "Reserved",                                                     // 10
        "Reserved",                                                     // 11
        "Reserved",                                                     // 12
        "Reserved",                                                     // 13
        "Reserved",                                                     // 14
        "Reserved",                                                     // 15
        "Coded slice segment of a BLA picture",                         // 16
        "Coded slice segment of a BLA picture",                         // 17
        "Coded slice segment of a BLA picture",                         // 18
        "Coded slice segment of an IDR picture",                        // 19
        "Coded slice segment of an IDR picture",                        // 20
        "Coded slice segment of a CRA picture",                         // 21
        "Reserved",                                                     // 22
        "Reserved",                                                     // 23
        "Reserved",                                                     // 24
        "Reserved",                                                     // 25
        "Reserved",                                                     // 26
        "Reserved",                                                     // 27
        "Reserved",                                                     // 28
        "Reserved",                                                     // 29
        "Reserved",                                                     // 30
        "Reserved",                                                     // 31
        "Video parameter set",                                          // 32
        "Sequence parameter set",                                       // 33
        "Picture parameter set",                                        // 34
        "Access unit delimiter",                                        // 35
        "End of sequence",                                              // 36
        "End of bitstream",                                             // 37
        "Filler data",                                                  // 38
        "Supplemental enhancement information (SEI)",                   // 39
        "Supplemental enhancement information (SEI)",                   // 40
        "Reserved",                                                     // 41
        "Reserved",                                                     // 42
        "Reserved",                                                     // 43
        "Reserved",                                                     // 44
        "Reserved",                                                     // 45
        "Reserved",                                                     // 46
        "Reserved",                                                     // 47
        "Unspecified",                                                  // 48
        "Unspecified",                                                  // 49
        "Unspecified",                                                  // 50
        "Unspecified",                                                  // 51
        "Unspecified",                                                  // 52
        "Unspecified",                                                  // 53
        "Unspecified",                                                  // 54
        "Unspecified",                                                  // 55
        "Unspecified",                                                  // 56
        "Unspecified",                                                  // 57
        "Unspecified",                                                  // 58
        "Unspecified",                                                  // 59
        "Unspecified",                                                  // 60
        "Unspecified",                                                  // 61
        "Unspecified",                                                  // 62
        "Unspecified",                                                  // 63
};
#endif

#ifdef DEBUG
static unsigned numDebugTabs = 1;
#define DEBUG_PRINT_TABS \
    for (unsigned _i = 0; _i < numDebugTabs; ++_i) fprintf(stderr, "\t")
#define DEBUG_PRINT(x)                      \
    do {                                    \
        DEBUG_PRINT_TABS;                   \
        fprintf(stderr, "%s: %d\n", #x, x); \
    } while (0)
#define DEBUG_STR(x)                \
    do {                            \
        DEBUG_PRINT_TABS;           \
        fprintf(stderr, "%s\n", x); \
    } while (0)
class DebugTab {
public:
    DebugTab() { ++numDebugTabs; }
    ~DebugTab() { --numDebugTabs; }
};
#define DEBUG_TAB DebugTab dummy
#else
#define DEBUG_PRINT(x) \
    do {               \
        x = x;         \
    } while (0)
// Note: the "x=x;" statement is intended to eliminate "unused variable"
// compiler warning messages
#define DEBUG_STR(x) \
    do {             \
    } while (0)
#define DEBUG_TAB \
    do {          \
    } while (0)
#endif

void H264or5VideoStreamParser::profile_tier_level(
        BitVector& bv, unsigned max_sub_layers_minus1) {
    bv.skipBits(96);

    unsigned i;
    Boolean sub_layer_profile_present_flag[7], sub_layer_level_present_flag[7];
    for (i = 0; i < max_sub_layers_minus1; ++i) {
        sub_layer_profile_present_flag[i] = bv.get1BitBoolean();
        sub_layer_level_present_flag[i] = bv.get1BitBoolean();
    }
    if (max_sub_layers_minus1 > 0) {
        bv.skipBits(2 * (8 - max_sub_layers_minus1));  // reserved_zero_2bits
    }
    for (i = 0; i < max_sub_layers_minus1; ++i) {
        if (sub_layer_profile_present_flag[i]) {
            bv.skipBits(88);
        }
        if (sub_layer_level_present_flag[i]) {
            bv.skipBits(8);  // sub_layer_level_idc[i]
        }
    }
}

void H264or5VideoStreamParser ::analyze_vui_parameters(
        BitVector& bv, unsigned& num_units_in_tick, unsigned& time_scale) {
    Boolean aspect_ratio_info_present_flag = bv.get1BitBoolean();
    DEBUG_PRINT(aspect_ratio_info_present_flag);
    if (aspect_ratio_info_present_flag) {
        DEBUG_TAB;
        unsigned aspect_ratio_idc = bv.getBits(8);
        DEBUG_PRINT(aspect_ratio_idc);
        if (aspect_ratio_idc == 255 /*Extended_SAR*/) {
            bv.skipBits(32);  // sar_width; sar_height
        }
    }
    Boolean overscan_info_present_flag = bv.get1BitBoolean();
    DEBUG_PRINT(overscan_info_present_flag);
    if (overscan_info_present_flag) {
        bv.skipBits(1);  // overscan_appropriate_flag
    }
    Boolean video_signal_type_present_flag = bv.get1BitBoolean();
    DEBUG_PRINT(video_signal_type_present_flag);
    if (video_signal_type_present_flag) {
        DEBUG_TAB;
        bv.skipBits(4);  // video_format; video_full_range_flag
        Boolean colour_description_present_flag = bv.get1BitBoolean();
        DEBUG_PRINT(colour_description_present_flag);
        if (colour_description_present_flag) {
            bv.skipBits(24);  // colour_primaries; transfer_characteristics;
                              // matrix_coefficients
        }
    }
    Boolean chroma_loc_info_present_flag = bv.get1BitBoolean();
    DEBUG_PRINT(chroma_loc_info_present_flag);
    if (chroma_loc_info_present_flag) {
        (void)bv.get_expGolomb();  // chroma_sample_loc_type_top_field
        (void)bv.get_expGolomb();  // chroma_sample_loc_type_bottom_field
    }
    if (fHNumber == 265) {
        bv.skipBits(2);  // neutral_chroma_indication_flag, field_seq_flag
        Boolean frame_field_info_present_flag = bv.get1BitBoolean();
        DEBUG_PRINT(frame_field_info_present_flag);
        pic_struct_present_flag =
                frame_field_info_present_flag;  // hack to make H.265 like H.264
        Boolean default_display_window_flag = bv.get1BitBoolean();
        DEBUG_PRINT(default_display_window_flag);
        if (default_display_window_flag) {
            (void)bv.get_expGolomb();  // def_disp_win_left_offset
            (void)bv.get_expGolomb();  // def_disp_win_right_offset
            (void)bv.get_expGolomb();  // def_disp_win_top_offset
            (void)bv.get_expGolomb();  // def_disp_win_bottom_offset
        }
    }
    Boolean timing_info_present_flag = bv.get1BitBoolean();
    DEBUG_PRINT(timing_info_present_flag);
    if (timing_info_present_flag) {
        DEBUG_TAB;
        num_units_in_tick = bv.getBits(32);
        DEBUG_PRINT(num_units_in_tick);
        time_scale = bv.getBits(32);
        DEBUG_PRINT(time_scale);
        if (fHNumber == 264) {
            Boolean fixed_frame_rate_flag = bv.get1BitBoolean();
            DEBUG_PRINT(fixed_frame_rate_flag);
        } else {  // 265
            Boolean vui_poc_proportional_to_timing_flag = bv.get1BitBoolean();
            DEBUG_PRINT(vui_poc_proportional_to_timing_flag);
            if (vui_poc_proportional_to_timing_flag) {
                unsigned vui_num_ticks_poc_diff_one_minus1 = bv.get_expGolomb();
                DEBUG_PRINT(vui_num_ticks_poc_diff_one_minus1);
            }
            return;  // For H.265, don't bother parsing any more of this #####
        }
    }
    // The following is H.264 only: #####
    Boolean nal_hrd_parameters_present_flag = bv.get1BitBoolean();
    DEBUG_PRINT(nal_hrd_parameters_present_flag);
    if (nal_hrd_parameters_present_flag) analyze_hrd_parameters(bv);
    Boolean vcl_hrd_parameters_present_flag = bv.get1BitBoolean();
    DEBUG_PRINT(vcl_hrd_parameters_present_flag);
    if (vcl_hrd_parameters_present_flag) analyze_hrd_parameters(bv);
    CpbDpbDelaysPresentFlag =
            nal_hrd_parameters_present_flag || vcl_hrd_parameters_present_flag;
    if (CpbDpbDelaysPresentFlag) {
        bv.skipBits(1);  // low_delay_hrd_flag
    }
    pic_struct_present_flag = bv.get1BitBoolean();
    DEBUG_PRINT(pic_struct_present_flag);
}

void H264or5VideoStreamParser::analyze_hrd_parameters(BitVector& bv) {
    DEBUG_TAB;
    unsigned cpb_cnt_minus1 = bv.get_expGolomb();
    DEBUG_PRINT(cpb_cnt_minus1);
    unsigned bit_rate_scale = bv.getBits(4);
    DEBUG_PRINT(bit_rate_scale);
    unsigned cpb_size_scale = bv.getBits(4);
    DEBUG_PRINT(cpb_size_scale);
    for (unsigned SchedSelIdx = 0; SchedSelIdx <= cpb_cnt_minus1;
         ++SchedSelIdx) {
        DEBUG_TAB;
        DEBUG_PRINT(SchedSelIdx);
        unsigned bit_rate_value_minus1 = bv.get_expGolomb();
        DEBUG_PRINT(bit_rate_value_minus1);
        unsigned cpb_size_value_minus1 = bv.get_expGolomb();
        DEBUG_PRINT(cpb_size_value_minus1);
        Boolean cbr_flag = bv.get1BitBoolean();
        DEBUG_PRINT(cbr_flag);
    }
    unsigned initial_cpb_removal_delay_length_minus1 = bv.getBits(5);
    DEBUG_PRINT(initial_cpb_removal_delay_length_minus1);
    cpb_removal_delay_length_minus1 = bv.getBits(5);
    DEBUG_PRINT(cpb_removal_delay_length_minus1);
    dpb_output_delay_length_minus1 = bv.getBits(5);
    DEBUG_PRINT(dpb_output_delay_length_minus1);
    unsigned time_offset_length = bv.getBits(5);
    DEBUG_PRINT(time_offset_length);
}

void H264or5VideoStreamParser ::analyze_video_parameter_set_data(
        unsigned& num_units_in_tick, unsigned& time_scale) {
    num_units_in_tick = time_scale = 0;  // default values

    // Begin by making a copy of the NAL unit data, removing any 'emulation
    // prevention' bytes:
    u_int8_t vps[VPS_MAX_SIZE];
    unsigned vpsSize;
    removeEmulationBytes(vps, sizeof vps, vpsSize);

    BitVector bv(vps, 0, 8 * vpsSize);

    // Assert: fHNumber == 265 (because this function is called only when
    // parsing H.265)
    unsigned i;

    bv.skipBits(28);  // nal_unit_header, vps_video_parameter_set_id,
                      // vps_reserved_three_2bits, vps_max_layers_minus1
    unsigned vps_max_sub_layers_minus1 = bv.getBits(3);
    DEBUG_PRINT(vps_max_sub_layers_minus1);
    bv.skipBits(
            17);  // vps_temporal_id_nesting_flag, vps_reserved_0xffff_16bits
    profile_tier_level(bv, vps_max_sub_layers_minus1);
    Boolean vps_sub_layer_ordering_info_present_flag = bv.get1BitBoolean();
    DEBUG_PRINT(vps_sub_layer_ordering_info_present_flag);
    for (i = vps_sub_layer_ordering_info_present_flag
                     ? 0
                     : vps_max_sub_layers_minus1;
         i <= vps_max_sub_layers_minus1; ++i) {
        (void)bv.get_expGolomb();  // vps_max_dec_pic_buffering_minus1[i]
        (void)bv.get_expGolomb();  // vps_max_num_reorder_pics[i]
        (void)bv.get_expGolomb();  // vps_max_latency_increase_plus1[i]
    }
    unsigned vps_max_layer_id = bv.getBits(6);
    DEBUG_PRINT(vps_max_layer_id);
    unsigned vps_num_layer_sets_minus1 = bv.get_expGolomb();
    DEBUG_PRINT(vps_num_layer_sets_minus1);
    for (i = 1; i <= vps_num_layer_sets_minus1; ++i) {
        bv.skipBits(vps_max_layer_id +
                    1);  // layer_id_included_flag[i][0..vps_max_layer_id]
    }
    Boolean vps_timing_info_present_flag = bv.get1BitBoolean();
    DEBUG_PRINT(vps_timing_info_present_flag);
    if (vps_timing_info_present_flag) {
        DEBUG_TAB;
        num_units_in_tick = bv.getBits(32);
        DEBUG_PRINT(num_units_in_tick);
        time_scale = bv.getBits(32);
        DEBUG_PRINT(time_scale);
        Boolean vps_poc_proportional_to_timing_flag = bv.get1BitBoolean();
        DEBUG_PRINT(vps_poc_proportional_to_timing_flag);
        if (vps_poc_proportional_to_timing_flag) {
            unsigned vps_num_ticks_poc_diff_one_minus1 = bv.get_expGolomb();
            DEBUG_PRINT(vps_num_ticks_poc_diff_one_minus1);
        }
    }
    Boolean vps_extension_flag = bv.get1BitBoolean();
    DEBUG_PRINT(vps_extension_flag);
}

void H264or5VideoStreamParser ::analyze_seq_parameter_set_data(
        unsigned& num_units_in_tick, unsigned& time_scale) {
    num_units_in_tick = time_scale = 0;  // default values

    // Begin by making a copy of the NAL unit data, removing any 'emulation
    // prevention' bytes:
    u_int8_t sps[SPS_MAX_SIZE];
    unsigned spsSize;
    removeEmulationBytes(sps, sizeof sps, spsSize);

    BitVector bv(sps, 0, 8 * spsSize);

    if (fHNumber == 264) {
        bv.skipBits(8);  // forbidden_zero_bit; nal_ref_idc; nal_unit_type
        unsigned profile_idc = bv.getBits(8);
        DEBUG_PRINT(profile_idc);
        unsigned constraint_setN_flag =
                bv.getBits(8);  // also "reserved_zero_2bits" at end
        DEBUG_PRINT(constraint_setN_flag);
        unsigned level_idc = bv.getBits(8);
        DEBUG_PRINT(level_idc);
        unsigned seq_parameter_set_id = bv.get_expGolomb();
        DEBUG_PRINT(seq_parameter_set_id);
        if (profile_idc == 100 || profile_idc == 110 || profile_idc == 122 ||
            profile_idc == 244 || profile_idc == 44 || profile_idc == 83 ||
            profile_idc == 86 || profile_idc == 118 || profile_idc == 128) {
            DEBUG_TAB;
            unsigned chroma_format_idc = bv.get_expGolomb();
            DEBUG_PRINT(chroma_format_idc);
            if (chroma_format_idc == 3) {
                DEBUG_TAB;
                Boolean separate_colour_plane_flag = bv.get1BitBoolean();
                DEBUG_PRINT(separate_colour_plane_flag);
            }
            (void)bv.get_expGolomb();  // bit_depth_luma_minus8
            (void)bv.get_expGolomb();  // bit_depth_chroma_minus8
            bv.skipBits(1);            // qpprime_y_zero_transform_bypass_flag
            Boolean seq_scaling_matrix_present_flag = bv.get1BitBoolean();
            DEBUG_PRINT(seq_scaling_matrix_present_flag);
            if (seq_scaling_matrix_present_flag) {
                for (int i = 0; i < ((chroma_format_idc != 3) ? 8 : 12); ++i) {
                    DEBUG_TAB;
                    DEBUG_PRINT(i);
                    Boolean seq_scaling_list_present_flag = bv.get1BitBoolean();
                    DEBUG_PRINT(seq_scaling_list_present_flag);
                    if (seq_scaling_list_present_flag) {
                        DEBUG_TAB;
                        unsigned sizeOfScalingList = i < 6 ? 16 : 64;
                        unsigned lastScale = 8;
                        unsigned nextScale = 8;
                        for (unsigned j = 0; j < sizeOfScalingList; ++j) {
                            DEBUG_TAB;
                            DEBUG_PRINT(j);
                            DEBUG_PRINT(nextScale);
                            if (nextScale != 0) {
                                DEBUG_TAB;
                                int delta_scale = bv.get_expGolombSigned();
                                DEBUG_PRINT(delta_scale);
                                nextScale =
                                        (lastScale + delta_scale + 256) % 256;
                            }
                            lastScale =
                                    (nextScale == 0) ? lastScale : nextScale;
                            DEBUG_PRINT(lastScale);
                        }
                    }
                }
            }
        }
        unsigned log2_max_frame_num_minus4 = bv.get_expGolomb();
        DEBUG_PRINT(log2_max_frame_num_minus4);
        unsigned pic_order_cnt_type = bv.get_expGolomb();
        DEBUG_PRINT(pic_order_cnt_type);
        if (pic_order_cnt_type == 0) {
            DEBUG_TAB;
            unsigned log2_max_pic_order_cnt_lsb_minus4 = bv.get_expGolomb();
            DEBUG_PRINT(log2_max_pic_order_cnt_lsb_minus4);
        } else if (pic_order_cnt_type == 1) {
            DEBUG_TAB;
            bv.skipBits(1);                  // delta_pic_order_always_zero_flag
            (void)bv.get_expGolombSigned();  // offset_for_non_ref_pic
            (void)bv.get_expGolombSigned();  // offset_for_top_to_bottom_field
            unsigned num_ref_frames_in_pic_order_cnt_cycle = bv.get_expGolomb();
            DEBUG_PRINT(num_ref_frames_in_pic_order_cnt_cycle);
            for (unsigned i = 0; i < num_ref_frames_in_pic_order_cnt_cycle;
                 ++i) {
                (void)bv.get_expGolombSigned();  // offset_for_ref_frame[i]
            }
        }
        unsigned max_num_ref_frames = bv.get_expGolomb();
        DEBUG_PRINT(max_num_ref_frames);
        Boolean gaps_in_frame_num_value_allowed_flag = bv.get1BitBoolean();
        DEBUG_PRINT(gaps_in_frame_num_value_allowed_flag);
        unsigned pic_width_in_mbs_minus1 = bv.get_expGolomb();
        DEBUG_PRINT(pic_width_in_mbs_minus1);
        unsigned pic_height_in_map_units_minus1 = bv.get_expGolomb();
        DEBUG_PRINT(pic_height_in_map_units_minus1);
        Boolean frame_mbs_only_flag = bv.get1BitBoolean();
        DEBUG_PRINT(frame_mbs_only_flag);
        if (!frame_mbs_only_flag) {
            bv.skipBits(1);  // mb_adaptive_frame_field_flag
        }
        bv.skipBits(1);  // direct_8x8_inference_flag
        Boolean frame_cropping_flag = bv.get1BitBoolean();
        DEBUG_PRINT(frame_cropping_flag);
        if (frame_cropping_flag) {
            (void)bv.get_expGolomb();  // frame_crop_left_offset
            (void)bv.get_expGolomb();  // frame_crop_right_offset
            (void)bv.get_expGolomb();  // frame_crop_top_offset
            (void)bv.get_expGolomb();  // frame_crop_bottom_offset
        }
        Boolean vui_parameters_present_flag = bv.get1BitBoolean();
        DEBUG_PRINT(vui_parameters_present_flag);
        if (vui_parameters_present_flag) {
            DEBUG_TAB;
            analyze_vui_parameters(bv, num_units_in_tick, time_scale);
        }
    } else {  // 265
        unsigned i;

        bv.skipBits(16);  // nal_unit_header
        bv.skipBits(4);   // sps_video_parameter_set_id
        unsigned sps_max_sub_layers_minus1 = bv.getBits(3);
        DEBUG_PRINT(sps_max_sub_layers_minus1);
        bv.skipBits(1);  // sps_temporal_id_nesting_flag
        profile_tier_level(bv, sps_max_sub_layers_minus1);
        (void)bv.get_expGolomb();  // sps_seq_parameter_set_id
        unsigned chroma_format_idc = bv.get_expGolomb();
        DEBUG_PRINT(chroma_format_idc);
        if (chroma_format_idc == 3)
            bv.skipBits(1);  // separate_colour_plane_flag
        unsigned pic_width_in_luma_samples = bv.get_expGolomb();
        DEBUG_PRINT(pic_width_in_luma_samples);
        unsigned pic_height_in_luma_samples = bv.get_expGolomb();
        DEBUG_PRINT(pic_height_in_luma_samples);
        Boolean conformance_window_flag = bv.get1BitBoolean();
        DEBUG_PRINT(conformance_window_flag);
        if (conformance_window_flag) {
            DEBUG_TAB;
            unsigned conf_win_left_offset = bv.get_expGolomb();
            DEBUG_PRINT(conf_win_left_offset);
            unsigned conf_win_right_offset = bv.get_expGolomb();
            DEBUG_PRINT(conf_win_right_offset);
            unsigned conf_win_top_offset = bv.get_expGolomb();
            DEBUG_PRINT(conf_win_top_offset);
            unsigned conf_win_bottom_offset = bv.get_expGolomb();
            DEBUG_PRINT(conf_win_bottom_offset);
        }
        (void)bv.get_expGolomb();  // bit_depth_luma_minus8
        (void)bv.get_expGolomb();  // bit_depth_chroma_minus8
        unsigned log2_max_pic_order_cnt_lsb_minus4 = bv.get_expGolomb();
        Boolean sps_sub_layer_ordering_info_present_flag = bv.get1BitBoolean();
        DEBUG_PRINT(sps_sub_layer_ordering_info_present_flag);
        for (i = (sps_sub_layer_ordering_info_present_flag
                          ? 0
                          : sps_max_sub_layers_minus1);
             i <= sps_max_sub_layers_minus1; ++i) {
            (void)bv.get_expGolomb();  // sps_max_dec_pic_buffering_minus1[i]
            (void)bv.get_expGolomb();  // sps_max_num_reorder_pics[i]
            (void)bv.get_expGolomb();  // sps_max_latency_increase[i]
        }
        (void)bv.get_expGolomb();  // log2_min_luma_coding_block_size_minus3
        (void)bv.get_expGolomb();  // log2_diff_max_min_luma_coding_block_size
        (void)bv.get_expGolomb();  // log2_min_transform_block_size_minus2
        (void)bv.get_expGolomb();  // log2_diff_max_min_transform_block_size
        (void)bv.get_expGolomb();  // max_transform_hierarchy_depth_inter
        (void)bv.get_expGolomb();  // max_transform_hierarchy_depth_intra
        Boolean scaling_list_enabled_flag = bv.get1BitBoolean();
        DEBUG_PRINT(scaling_list_enabled_flag);
        if (scaling_list_enabled_flag) {
            DEBUG_TAB;
            Boolean sps_scaling_list_data_present_flag = bv.get1BitBoolean();
            DEBUG_PRINT(sps_scaling_list_data_present_flag);
            if (sps_scaling_list_data_present_flag) {
                // scaling_list_data()
                DEBUG_TAB;
                for (unsigned sizeId = 0; sizeId < 4; ++sizeId) {
                    DEBUG_PRINT(sizeId);
                    for (unsigned matrixId = 0;
                         matrixId < (sizeId == 3 ? 2 : 6); ++matrixId) {
                        DEBUG_TAB;
                        DEBUG_PRINT(matrixId);
                        Boolean scaling_list_pred_mode_flag =
                                bv.get1BitBoolean();
                        DEBUG_PRINT(scaling_list_pred_mode_flag);
                        if (!scaling_list_pred_mode_flag) {
                            (void)bv.get_expGolomb();  // scaling_list_pred_matrix_id_delta[sizeId][matrixId]
                        } else {
                            unsigned const c = 1 << (4 + (sizeId << 1));
                            unsigned coefNum = c < 64 ? c : 64;
                            if (sizeId > 1) {
                                (void)bv.get_expGolomb();  // scaling_list_dc_coef_minus8[sizeId][matrixId]
                            }
                            for (i = 0; i < coefNum; ++i) {
                                (void)bv.get_expGolomb();  // scaling_list_delta_coef
                            }
                        }
                    }
                }
            }
        }
        bv.skipBits(
                2);  // amp_enabled_flag, sample_adaptive_offset_enabled_flag
        Boolean pcm_enabled_flag = bv.get1BitBoolean();
        DEBUG_PRINT(pcm_enabled_flag);
        if (pcm_enabled_flag) {
            bv.skipBits(8);            // pcm_sample_bit_depth_luma_minus1,
                                       // pcm_sample_bit_depth_chroma_minus1
            (void)bv.get_expGolomb();  // log2_min_pcm_luma_coding_block_size_minus3
            (void)bv.get_expGolomb();  // log2_diff_max_min_pcm_luma_coding_block_size
            bv.skipBits(1);            // pcm_loop_filter_disabled_flag
        }
        unsigned num_short_term_ref_pic_sets = bv.get_expGolomb();
        DEBUG_PRINT(num_short_term_ref_pic_sets);
        unsigned num_negative_pics = 0, prev_num_negative_pics = 0;
        unsigned num_positive_pics = 0, prev_num_positive_pics = 0;
        for (i = 0; i < num_short_term_ref_pic_sets; ++i) {
            // short_term_ref_pic_set(i):
            DEBUG_TAB;
            DEBUG_PRINT(i);
            Boolean inter_ref_pic_set_prediction_flag = False;
            if (i != 0) {
                inter_ref_pic_set_prediction_flag = bv.get1BitBoolean();
            }
            DEBUG_PRINT(inter_ref_pic_set_prediction_flag);
            if (inter_ref_pic_set_prediction_flag) {
                DEBUG_TAB;
                if (i == num_short_term_ref_pic_sets) {
                    // This can't happen here, but it's in the spec, so we
                    // include it for completeness
                    (void)bv.get_expGolomb();  // delta_idx_minus1
                }
                bv.skipBits(1);            // delta_rps_sign
                (void)bv.get_expGolomb();  // abs_delta_rps_minus1
                unsigned NumDeltaPocs = prev_num_negative_pics +
                                        prev_num_positive_pics;  // correct???
                for (unsigned j = 0; j < NumDeltaPocs; ++j) {
                    DEBUG_PRINT(j);
                    Boolean used_by_curr_pic_flag = bv.get1BitBoolean();
                    DEBUG_PRINT(used_by_curr_pic_flag);
                    if (!used_by_curr_pic_flag)
                        bv.skipBits(1);  // use_delta_flag[j]
                }
            } else {
                prev_num_negative_pics = num_negative_pics;
                num_negative_pics = bv.get_expGolomb();
                DEBUG_PRINT(num_negative_pics);
                prev_num_positive_pics = num_positive_pics;
                num_positive_pics = bv.get_expGolomb();
                DEBUG_PRINT(num_positive_pics);
                unsigned k;
                for (k = 0; k < num_negative_pics; ++k) {
                    (void)bv.get_expGolomb();  // delta_poc_s0_minus1[k]
                    bv.skipBits(1);            // used_by_curr_pic_s0_flag[k]
                }
                for (k = 0; k < num_positive_pics; ++k) {
                    (void)bv.get_expGolomb();  // delta_poc_s1_minus1[k]
                    bv.skipBits(1);            // used_by_curr_pic_s1_flag[k]
                }
            }
        }
        Boolean long_term_ref_pics_present_flag = bv.get1BitBoolean();
        DEBUG_PRINT(long_term_ref_pics_present_flag);
        if (long_term_ref_pics_present_flag) {
            DEBUG_TAB;
            unsigned num_long_term_ref_pics_sps = bv.get_expGolomb();
            DEBUG_PRINT(num_long_term_ref_pics_sps);
            for (i = 0; i < num_long_term_ref_pics_sps; ++i) {
                bv.skipBits(
                        log2_max_pic_order_cnt_lsb_minus4);  // lt_ref_pic_poc_lsb_sps[i]
                bv.skipBits(1);  // used_by_curr_pic_lt_sps_flag[1]
            }
        }
        bv.skipBits(2);  // sps_temporal_mvp_enabled_flag,
                         // strong_intra_smoothing_enabled_flag
        Boolean vui_parameters_present_flag = bv.get1BitBoolean();
        DEBUG_PRINT(vui_parameters_present_flag);
        if (vui_parameters_present_flag) {
            DEBUG_TAB;
            analyze_vui_parameters(bv, num_units_in_tick, time_scale);
        }
        Boolean sps_extension_flag = bv.get1BitBoolean();
        DEBUG_PRINT(sps_extension_flag);
    }
}

#define SEI_MAX_SIZE 5000  // larger than the largest possible SEI NAL unit

#ifdef DEBUG
#define MAX_SEI_PAYLOAD_TYPE_DESCRIPTION_H264 46
char const*
        sei_payloadType_description_h264[MAX_SEI_PAYLOAD_TYPE_DESCRIPTION_H264 +
                                         1] = {
                "buffering_period",                      // 0
                "pic_timing",                            // 1
                "pan_scan_rect",                         // 2
                "filler_payload",                        // 3
                "user_data_registered_itu_t_t35",        // 4
                "user_data_unregistered",                // 5
                "recovery_point",                        // 6
                "dec_ref_pic_marking_repetition",        // 7
                "spare_pic",                             // 8
                "scene_info",                            // 9
                "sub_seq_info",                          // 10
                "sub_seq_layer_characteristics",         // 11
                "sub_seq_characteristics",               // 12
                "full_frame_freeze",                     // 13
                "full_frame_freeze_release",             // 14
                "full_frame_snapshot",                   // 15
                "progressive_refinement_segment_start",  // 16
                "progressive_refinement_segment_end",    // 17
                "motion_constrained_slice_group_set",    // 18
                "film_grain_characteristics",            // 19
                "deblocking_filter_display_preference",  // 20
                "stereo_video_info",                     // 21
                "post_filter_hint",                      // 22
                "tone_mapping_info",                     // 23
                "scalability_info",                      // 24
                "sub_pic_scalable_layer",                // 25
                "non_required_layer_rep",                // 26
                "priority_layer_info",                   // 27
                "layers_not_present",                    // 28
                "layer_dependency_change",               // 29
                "scalable_nesting",                      // 30
                "base_layer_temporal_hrd",               // 31
                "quality_layer_integrity_check",         // 32
                "redundant_pic_property",                // 33
                "tl0_dep_rep_index",                     // 34
                "tl_switching_point",                    // 35
                "parallel_decoding_info",                // 36
                "mvc_scalable_nesting",                  // 37
                "view_scalability_info",                 // 38
                "multiview_scene_info",                  // 39
                "multiview_acquisition_info",            // 40
                "non_required_view_component",           // 41
                "view_dependency_change",                // 42
                "operation_points_not_present",          // 43
                "base_view_temporal_hrd",                // 44
                "frame_packing_arrangement",             // 45
                "reserved_sei_message"                   // 46 or higher
};
#endif

void H264or5VideoStreamParser::analyze_sei_data(u_int8_t nal_unit_type) {
    // Begin by making a copy of the NAL unit data, removing any 'emulation
    // prevention' bytes:
    u_int8_t sei[SEI_MAX_SIZE];
    unsigned seiSize;
    removeEmulationBytes(sei, sizeof sei, seiSize);

    unsigned j = 1;  // skip the initial byte (forbidden_zero_bit; nal_ref_idc;
                     // nal_unit_type); we've already seen it
    while (j < seiSize) {
        unsigned payloadType = 0;
        do {
            payloadType += sei[j];
        } while (sei[j++] == 255 && j < seiSize);
        if (j >= seiSize) break;

        unsigned payloadSize = 0;
        do {
            payloadSize += sei[j];
        } while (sei[j++] == 255 && j < seiSize);
        if (j >= seiSize) break;

#ifdef DEBUG
        char const* description;
        if (fHNumber == 264) {
            unsigned descriptionNum =
                    payloadType <= MAX_SEI_PAYLOAD_TYPE_DESCRIPTION_H264
                            ? payloadType
                            : MAX_SEI_PAYLOAD_TYPE_DESCRIPTION_H264;
            description = sei_payloadType_description_h264[descriptionNum];
        } else {  // 265
            description = payloadType == 3
                                  ? "filler_payload"
                                  : payloadType == 4
                                            ? "user_data_registered_itu_t_t35"
                                            : payloadType == 5
                                                      ? "user_data_unregistered"
                                                      : payloadType == 17
                                                                ? "progressive_"
                                                                  "refinement_"
                                                                  "segment_end"
                                                                : payloadType == 22
                                                                          ? "po"
                                                                            "st"
                                                                            "_f"
                                                                            "il"
                                                                            "te"
                                                                            "r_"
                                                                            "hi"
                                                                            "nt"
                                                                          : (payloadType ==
                                                                                     132 &&
                                                                             nal_unit_type ==
                                                                                     SUFFIX_SEI_NUT)
                                                                                    ? "decoded_picture_hash"
                                                                                    : nal_unit_type ==
                                                                                                      SUFFIX_SEI_NUT
                                                                                              ? "reserved_sei_message"
                                                                                              : payloadType == 0
                                                                                                        ? "buffering_period"
                                                                                                        : payloadType == 1 ? "pic_timing"
                                                                                                                           : payloadType == 2 ? "pan_scan_rect"
                                                                                                                                              : payloadType == 6 ? "recovery_point" : payloadType == 9 ? "scene_info" : payloadType == 15 ? "picture_snapshot" : payloadType == 16 ? "progressive_refinement_segment_start" : payloadType == 19 ? "film_grain_characteristics" : payloadType == 23 ? "tone_mapping_info" : payloadType == 45 ? "frame_packing_arrangement" : payloadType == 47 ? "display_orientation" : payloadType == 128 ? "structure_of_pictures_info" : payloadType == 129 ? "active_parameter_sets" : payloadType == 130 ? "decoding_unit_info" : payloadType == 131 ? "temporal_sub_layer_zero_index" : payloadType == 133 ? "scalable_nesting" : payloadType == 134 ? "region_refresh_info" : "reserved_sei_message";
        }
        fprintf(stderr, "\tpayloadType %d (\"%s\"); payloadSize %d\n",
                payloadType, description, payloadSize);
#endif

        analyze_sei_payload(payloadType, payloadSize, &sei[j]);
        j += payloadSize;
    }
}

void H264or5VideoStreamParser ::analyze_sei_payload(unsigned payloadType,
                                                    unsigned payloadSize,
                                                    u_int8_t* payload) {
    if (payloadType == 1 /* pic_timing, for both H.264 and H.265 */) {
        BitVector bv(payload, 0, 8 * payloadSize);

        DEBUG_TAB;
        if (CpbDpbDelaysPresentFlag) {
            unsigned cpb_removal_delay =
                    bv.getBits(cpb_removal_delay_length_minus1 + 1);
            DEBUG_PRINT(cpb_removal_delay);
            unsigned dpb_output_delay =
                    bv.getBits(dpb_output_delay_length_minus1 + 1);
            DEBUG_PRINT(dpb_output_delay);
        }
        double prevDeltaTfiDivisor = DeltaTfiDivisor;
        if (pic_struct_present_flag) {
            unsigned pic_struct = bv.getBits(4);
            DEBUG_PRINT(pic_struct);
            // Use this to set "DeltaTfiDivisor" (which is used to compute the
            // frame rate):
            if (fHNumber == 264) {
                DeltaTfiDivisor =
                        pic_struct == 0
                                ? 2.0
                                : pic_struct <= 2
                                          ? 1.0
                                          : pic_struct <= 4
                                                    ? 2.0
                                                    : pic_struct <= 6
                                                              ? 3.0
                                                              : pic_struct == 7
                                                                        ? 4.0
                                                                        : pic_struct == 8
                                                                                  ? 6.0
                                                                                  : 2.0;
            } else {  // H.265
                DeltaTfiDivisor =
                        pic_struct == 0
                                ? 2.0
                                : pic_struct <= 2
                                          ? 1.0
                                          : pic_struct <= 4
                                                    ? 2.0
                                                    : pic_struct <= 6
                                                              ? 3.0
                                                              : pic_struct == 7
                                                                        ? 2.0
                                                                        : pic_struct == 8
                                                                                  ? 3.0
                                                                                  : pic_struct <= 12
                                                                                            ? 1.0
                                                                                            : 2.0;
            }
        } else {
            if (fHNumber == 264) {
                // Need to get field_pic_flag from slice_header to set this
                // properly! #####
            } else {  // H.265
                DeltaTfiDivisor = 1.0;
            }
        }
        // If "DeltaTfiDivisor" has changed, and we've already computed the
        // frame rate, then adjust it, based on the new value of
        // "DeltaTfiDivisor":
        if (DeltaTfiDivisor != prevDeltaTfiDivisor && fParsedFrameRate != 0.0) {
            usingSource()->fFrameRate = fParsedFrameRate =
                    fParsedFrameRate * (prevDeltaTfiDivisor / DeltaTfiDivisor);
#ifdef DEBUG
            fprintf(stderr, "Changed frame rate to %f fps\n",
                    usingSource()->fFrameRate);
#endif
        }
        // Ignore the rest of the payload (timestamps) for now... #####
    }
}

void H264or5VideoStreamParser::flushInput() {
    fHaveSeenFirstStartCode = False;
    fHaveSeenFirstByteOfNALUnit = False;

    StreamParser::flushInput();
}

unsigned H264or5VideoStreamParser::parse() {
    try {
        // The stream must start with a 0x00000001:
        if (!fHaveSeenFirstStartCode) {
            // Skip over any input bytes that precede the first 0x00000001:
            u_int32_t first4Bytes;
            while ((first4Bytes = test4Bytes()) != 0x00000001) {
                get1Byte();
                setParseState();  // ensures that we progress over bad data
            }
            skipBytes(4);  // skip this initial code

            setParseState();
            fHaveSeenFirstStartCode = True;  // from now on
        }

        if (fOutputStartCodeSize > 0 && curFrameSize() == 0 && !haveSeenEOF()) {
            // Include a start code in the output:
            save4Bytes(0x00000001);
        }

        // Then save everything up until the next 0x00000001 (4 bytes) or
        // 0x000001 (3 bytes), or we hit EOF. Also make note of the first byte,
        // because it contains the "nal_unit_type":
        if (haveSeenEOF()) {
            // We hit EOF the last time that we tried to parse this data, so we
            // know that any remaining unparsed data forms a complete NAL unit,
            // and that there's no 'start code' at the end:
            unsigned remainingDataSize = totNumValidBytes() - curOffset();
#ifdef DEBUG
            unsigned const trailingNALUnitSize = remainingDataSize;
#endif
            while (remainingDataSize > 0) {
                u_int8_t nextByte = get1Byte();
                if (!fHaveSeenFirstByteOfNALUnit) {
                    fFirstByteOfNALUnit = nextByte;
                    fHaveSeenFirstByteOfNALUnit = True;
                }
                saveByte(nextByte);
                --remainingDataSize;
            }

#ifdef DEBUG
            if (fHNumber == 264) {
                u_int8_t nal_ref_idc = (fFirstByteOfNALUnit & 0x60) >> 5;
                u_int8_t nal_unit_type = fFirstByteOfNALUnit & 0x1F;
                fprintf(stderr,
                        "Parsed trailing %d-byte NAL-unit (nal_ref_idc: %d, "
                        "nal_unit_type: %d (\"%s\"))\n",
                        trailingNALUnitSize, nal_ref_idc, nal_unit_type,
                        nal_unit_type_description_h264[nal_unit_type]);
            } else {  // 265
                u_int8_t nal_unit_type = (fFirstByteOfNALUnit & 0x7E) >> 1;
                fprintf(stderr,
                        "Parsed trailing %d-byte NAL-unit (nal_unit_type: %d "
                        "(\"%s\"))\n",
                        trailingNALUnitSize, nal_unit_type,
                        nal_unit_type_description_h265[nal_unit_type]);
            }
#endif

            (void)get1Byte();  // forces another read, which will cause EOF to
                               // get handled for real this time
            return 0;
        } else {
            u_int32_t next4Bytes = test4Bytes();
            if (!fHaveSeenFirstByteOfNALUnit) {
                fFirstByteOfNALUnit = next4Bytes >> 24;
                fHaveSeenFirstByteOfNALUnit = True;
            }
            while (next4Bytes != 0x00000001 &&
                   (next4Bytes & 0xFFFFFF00) != 0x00000100) {
                // We save at least some of "next4Bytes".
                if ((unsigned)(next4Bytes & 0xFF) > 1) {
                    // Common case: 0x00000001 or 0x000001 definitely doesn't
                    // begin anywhere in "next4Bytes", so we save all of it:
                    save4Bytes(next4Bytes);
                    skipBytes(4);
                } else {
                    // Save the first byte, and continue testing the rest:
                    saveByte(next4Bytes >> 24);
                    skipBytes(1);
                }
                setParseState();  // ensures forward progress
                next4Bytes = test4Bytes();
            }
            // Assert: next4Bytes starts with 0x00000001 or 0x000001, and we've
            // saved all previous bytes (forming a complete NAL unit). Skip over
            // these remaining bytes, up until the start of the next NAL unit:
            if (next4Bytes == 0x00000001) {
                skipBytes(4);
            } else {
                skipBytes(3);
            }
        }

        fHaveSeenFirstByteOfNALUnit =
                False;  // for the next NAL unit that we'll parse
        u_int8_t nal_unit_type;
        if (fHNumber == 264) {
            nal_unit_type = fFirstByteOfNALUnit & 0x1F;
#ifdef DEBUG
            u_int8_t nal_ref_idc = (fFirstByteOfNALUnit & 0x60) >> 5;
            fprintf(stderr,
                    "Parsed %d-byte NAL-unit (nal_ref_idc: %d, nal_unit_type: "
                    "%d (\"%s\"))\n",
                    curFrameSize() - fOutputStartCodeSize, nal_ref_idc,
                    nal_unit_type,
                    nal_unit_type_description_h264[nal_unit_type]);
#endif
        } else {  // 265
            nal_unit_type = (fFirstByteOfNALUnit & 0x7E) >> 1;
#ifdef DEBUG
            fprintf(stderr,
                    "Parsed %d-byte NAL-unit (nal_unit_type: %d (\"%s\"))\n",
                    curFrameSize() - fOutputStartCodeSize, nal_unit_type,
                    nal_unit_type_description_h265[nal_unit_type]);
#endif
        }

        // Now that we have found (& copied) a NAL unit, process it if it's of
        // special interest to us:
        if (isVPS(nal_unit_type)) {  // Video parameter set
            // First, save a copy of this NAL unit, in case the downstream
            // object wants to see it:
            usingSource()->saveCopyOfVPS(fStartOfFrame + fOutputStartCodeSize,
                                         curFrameSize() - fOutputStartCodeSize);

            if (fParsedFrameRate == 0.0) {
                // We haven't yet parsed a frame rate from the stream.
                // So parse this NAL unit to check whether frame rate
                // information is present:
                unsigned num_units_in_tick, time_scale;
                analyze_video_parameter_set_data(num_units_in_tick, time_scale);
                if (time_scale > 0 && num_units_in_tick > 0) {
                    usingSource()->fFrameRate = fParsedFrameRate =
                            time_scale / (DeltaTfiDivisor * num_units_in_tick);
#ifdef DEBUG
                    fprintf(stderr, "Set frame rate to %f fps\n",
                            usingSource()->fFrameRate);
#endif
                } else {
#ifdef DEBUG
                    fprintf(stderr,
                            "\tThis \"Video Parameter Set\" NAL unit contained "
                            "no frame rate information, so we use a default "
                            "frame rate of %f fps\n",
                            usingSource()->fFrameRate);
#endif
                }
            }
        } else if (isSPS(nal_unit_type)) {  // Sequence parameter set
            // First, save a copy of this NAL unit, in case the downstream
            // object wants to see it:
            usingSource()->saveCopyOfSPS(fStartOfFrame + fOutputStartCodeSize,
                                         curFrameSize() - fOutputStartCodeSize);

            if (fParsedFrameRate == 0.0) {
                // We haven't yet parsed a frame rate from the stream.
                // So parse this NAL unit to check whether frame rate
                // information is present:
                unsigned num_units_in_tick, time_scale;
                analyze_seq_parameter_set_data(num_units_in_tick, time_scale);
                if (time_scale > 0 && num_units_in_tick > 0) {
                    usingSource()->fFrameRate = fParsedFrameRate =
                            time_scale / (DeltaTfiDivisor * num_units_in_tick);
#ifdef DEBUG
                    fprintf(stderr, "Set frame rate to %f fps\n",
                            usingSource()->fFrameRate);
#endif
                } else {
#ifdef DEBUG
                    fprintf(stderr,
                            "\tThis \"Sequence Parameter Set\" NAL unit "
                            "contained no frame rate information, so we use a "
                            "default frame rate of %f fps\n",
                            usingSource()->fFrameRate);
#endif
                }
            }
        } else if (isPPS(nal_unit_type)) {  // Picture parameter set
            // Save a copy of this NAL unit, in case the downstream object wants
            // to see it:
            usingSource()->saveCopyOfPPS(fStartOfFrame + fOutputStartCodeSize,
                                         curFrameSize() - fOutputStartCodeSize);
        } else if (isSEI(nal_unit_type)) {  // Supplemental enhancement
                                            // information (SEI)
            analyze_sei_data(nal_unit_type);
            // Later, perhaps adjust "fPresentationTime" if we saw a
            // "pic_timing" SEI payload??? #####
        }

        usingSource()->setPresentationTime();
#ifdef DEBUG
        unsigned long secs =
                (unsigned long)usingSource()->fPresentationTime.tv_sec;
        unsigned uSecs = (unsigned)usingSource()->fPresentationTime.tv_usec;
        fprintf(stderr, "\tPresentation time: %lu.%06u\n", secs, uSecs);
#endif

        // Now, check whether this NAL unit ends an 'access unit'.
        // (RTP streamers need to know this in order to figure out whether or
        // not to set the "M" bit.)
        Boolean thisNALUnitEndsAccessUnit;
        if (haveSeenEOF() || isEOF(nal_unit_type)) {
            // There is no next NAL unit, so we assume that this one ends the
            // current 'access unit':
            thisNALUnitEndsAccessUnit = True;
        } else if (usuallyBeginsAccessUnit(nal_unit_type)) {
            // These NAL units usually *begin* an access unit, so assume that
            // they don't end one here:
            thisNALUnitEndsAccessUnit = False;
        } else {
            // We need to check the *next* NAL unit to figure out whether
            // the current NAL unit ends an 'access unit':
            u_int8_t firstBytesOfNextNALUnit[3];
            testBytes(firstBytesOfNextNALUnit, 3);

            u_int8_t const& next_nal_unit_type =
                    fHNumber == 264
                            ? (firstBytesOfNextNALUnit[0] & 0x1F)
                            : ((firstBytesOfNextNALUnit[0] & 0x7E) >> 1);
            if (isVCL(next_nal_unit_type)) {
                // The high-order bit of the byte after the "nal_unit_header"
                // tells us whether it's the start of a new 'access unit' (and
                // thus the current NAL unit ends an 'access unit'):
                u_int8_t const byteAfter_nal_unit_header =
                        fHNumber == 264 ? firstBytesOfNextNALUnit[1]
                                        : firstBytesOfNextNALUnit[2];
                thisNALUnitEndsAccessUnit =
                        (byteAfter_nal_unit_header & 0x80) != 0;
            } else if (usuallyBeginsAccessUnit(next_nal_unit_type)) {
                // The next NAL unit's type is one that usually appears at the
                // start of an 'access unit', so we assume that the current NAL
                // unit ends an 'access unit':
                thisNALUnitEndsAccessUnit = True;
            } else {
                // The next NAL unit definitely doesn't start a new 'access
                // unit', which means that the current NAL unit doesn't end one:
                thisNALUnitEndsAccessUnit = False;
            }
        }

        if (thisNALUnitEndsAccessUnit) {
#ifdef DEBUG
            fprintf(stderr,
                    "*****This NAL unit ends the current access unit*****\n");
#endif
            usingSource()->fPictureEndMarker = True;
            ++usingSource()->fPictureCount;

            // Note that the presentation time for the next NAL unit will be
            // different:
            struct timeval& nextPT =
                    usingSource()->fNextPresentationTime;  // alias
            nextPT = usingSource()->fPresentationTime;
            double nextFraction =
                    nextPT.tv_usec / 1000000.0 + 1 / usingSource()->fFrameRate;
            unsigned nextSecsIncrement = (long)nextFraction;
            nextPT.tv_sec += (long)nextSecsIncrement;
            nextPT.tv_usec =
                    (long)((nextFraction - nextSecsIncrement) * 1000000);
        }
        setParseState();

        return curFrameSize();
    } catch (int /*e*/) {
#ifdef DEBUG
        fprintf(stderr,
                "H264or5VideoStreamParser::parse() EXCEPTION (This is normal "
                "behavior - *not* an error)\n");
#endif
        return 0;  // the parsing got interrupted
    }
}

unsigned removeH264or5EmulationBytes(u_int8_t* to,
                                     unsigned toMaxSize,
                                     u_int8_t const* from,
                                     unsigned fromSize) {
    unsigned toSize = 0;
    unsigned i = 0;
    while (i < fromSize && toSize + 1 < toMaxSize) {
        if (i + 2 < fromSize && from[i] == 0 && from[i + 1] == 0 &&
            from[i + 2] == 3) {
            to[toSize] = to[toSize + 1] = 0;
            toSize += 2;
            i += 3;
        } else {
            to[toSize] = from[i];
            toSize += 1;
            i += 1;
        }
    }

    return toSize;
}
