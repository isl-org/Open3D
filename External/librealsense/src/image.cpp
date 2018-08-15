// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015 Intel Corporation. All Rights Reserved.

#include "image.h"
#include "../include/librealsense/rsutil.h" // For projection/deprojection logic

#include <cstring> // For memcpy
#include <cmath>
#include <algorithm>

#include <tmmintrin.h> // For SSE3 intrinsics used in unpack_yuy2_sse

#pragma pack(push, 1) // All structs in this file are assumed to be byte-packed
namespace rsimpl
{

    ////////////////////////////
    // Image size computation //
    ////////////////////////////

    size_t get_image_size(int width, int height, rs_format format)
    {
        switch(format)
        {
        case RS_FORMAT_Z16: return width * height * 2;
        case RS_FORMAT_DISPARITY16: return width * height * 2;
        case RS_FORMAT_XYZ32F: return width * height * 12;
        case RS_FORMAT_YUYV: assert(width % 2 == 0); return width * height * 2;
        case RS_FORMAT_RGB8: return width * height * 3;
        case RS_FORMAT_BGR8: return width * height * 3;
        case RS_FORMAT_RGBA8: return width * height * 4;
        case RS_FORMAT_BGRA8: return width * height * 4;
        case RS_FORMAT_Y8: return width * height;
        case RS_FORMAT_Y16: return width * height * 2;
        case RS_FORMAT_RAW10: assert(width % 4 == 0); return width * 5/4 * height;
        default: assert(false); return 0;
        }    
    }

    //////////////////////////////
    // Naive unpacking routines //
    //////////////////////////////
    
    template<size_t SIZE> void copy_pixels(byte * const dest[], const byte * source, int count)
    {
        memcpy(dest[0], source, SIZE * count);
    }

    template<class SOURCE, class UNPACK> void unpack_pixels(byte * const dest[], int count, const SOURCE * source, UNPACK unpack)
    {
        auto out = reinterpret_cast<decltype(unpack(SOURCE())) *>(dest[0]);
        for(int i=0; i<count; ++i) *out++ = unpack(*source++);
    }

    void unpack_y16_from_y8    (byte * const d[], const byte * s, int n) { unpack_pixels(d, n, reinterpret_cast<const uint8_t  *>(s), [](uint8_t  pixel) -> uint16_t { return pixel | pixel << 8; }); }
    void unpack_y16_from_y16_10(byte * const d[], const byte * s, int n) { unpack_pixels(d, n, reinterpret_cast<const uint16_t *>(s), [](uint16_t pixel) -> uint16_t { return pixel << 6; }); }
    void unpack_y8_from_y16_10 (byte * const d[], const byte * s, int n) { unpack_pixels(d, n, reinterpret_cast<const uint16_t *>(s), [](uint16_t pixel) -> uint8_t  { return pixel >> 2; }); }

    /////////////////////////////
    // YUY2 unpacking routines //
    /////////////////////////////
    
    // This templated function unpacks YUY2 into Y8/Y16/RGB8/RGBA8/BGR8/BGRA8, depending on the compile-time parameter FORMAT.
    // It is expected that all branching outside of the loop control variable will be removed due to constant-folding.
    template<rs_format FORMAT> void unpack_yuy2(byte * const d [], const byte * s, int n)
    {
        assert(n % 16 == 0); // All currently supported color resolutions are multiples of 16 pixels. Could easily extend support to other resolutions by copying final n<16 pixels into a zero-padded buffer and recursively calling self for final iteration.
        auto src = reinterpret_cast<const __m128i *>(s);
        auto dst = reinterpret_cast<__m128i *>(d[0]);
        for(; n; n -= 16)
        {
            const __m128i zero = _mm_set1_epi8(0);
            const __m128i n100 = _mm_set1_epi16(100 << 4);
            const __m128i n208 = _mm_set1_epi16(208 << 4);
            const __m128i n298 = _mm_set1_epi16(298 << 4);
            const __m128i n409 = _mm_set1_epi16(409 << 4);
            const __m128i n516 = _mm_set1_epi16(516 << 4);
            const __m128i evens_odds = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);            

            // Load 8 YUY2 pixels each into two 16-byte registers
            __m128i s0 = _mm_loadu_si128(src++);
            __m128i s1 = _mm_loadu_si128(src++);

            if(FORMAT == RS_FORMAT_Y8)
            {   
                // Align all Y components and output 16 pixels (16 bytes) at once
                __m128i y0 = _mm_shuffle_epi8(s0, _mm_setr_epi8(1, 3, 5, 7, 9, 11, 13, 15,   0, 2, 4, 6, 8, 10, 12, 14));
                __m128i y1 = _mm_shuffle_epi8(s1, _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14,   1, 3, 5, 7, 9, 11, 13, 15));
                _mm_storeu_si128(dst++, _mm_alignr_epi8(y0, y1, 8));
                continue;
            }

            // Shuffle all Y components to the low order bytes of the register, and all U/V components to the high order bytes
            const __m128i evens_odd1s_odd3s = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 5, 9, 13, 3, 7, 11, 15); // to get yyyyyyyyuuuuvvvv
            __m128i yyyyyyyyuuuuvvvv0 = _mm_shuffle_epi8(s0, evens_odd1s_odd3s);
            __m128i yyyyyyyyuuuuvvvv8 = _mm_shuffle_epi8(s1, evens_odd1s_odd3s);

            // Retrieve all 16 Y components as 16-bit values (8 components per register))
            __m128i y16__0_7 = _mm_unpacklo_epi8(yyyyyyyyuuuuvvvv0, zero);         // convert to 16 bit
            __m128i y16__8_F = _mm_unpacklo_epi8(yyyyyyyyuuuuvvvv8, zero);         // convert to 16 bit

            if(FORMAT == RS_FORMAT_Y16)
            {      
                // Output 16 pixels (32 bytes) at once
                _mm_storeu_si128(dst++, _mm_slli_epi16(y16__0_7, 8));
                _mm_storeu_si128(dst++, _mm_slli_epi16(y16__8_F, 8));
                continue;
            }

            // Retrieve all 16 U and V components as 16-bit values (8 components per register)
            __m128i uv = _mm_unpackhi_epi32(yyyyyyyyuuuuvvvv0, yyyyyyyyuuuuvvvv8); // uuuuuuuuvvvvvvvv
            __m128i u = _mm_unpacklo_epi8(uv, uv);                                 //  uu uu uu uu uu uu uu uu  u's duplicated
            __m128i v = _mm_unpackhi_epi8(uv, uv);                                 //  vv vv vv vv vv vv vv vv            
            __m128i u16__0_7 = _mm_unpacklo_epi8(u, zero);                         // convert to 16 bit
            __m128i u16__8_F = _mm_unpackhi_epi8(u, zero);                         // convert to 16 bit
            __m128i v16__0_7 = _mm_unpacklo_epi8(v, zero);                         // convert to 16 bit
            __m128i v16__8_F = _mm_unpackhi_epi8(v, zero);                         // convert to 16 bit

            // Compute R, G, B values for first 8 pixels
            __m128i c16__0_7 = _mm_slli_epi16(_mm_subs_epi16(y16__0_7, _mm_set1_epi16(16)), 4);
            __m128i d16__0_7 = _mm_slli_epi16(_mm_subs_epi16(u16__0_7, _mm_set1_epi16(128)), 4); // perhaps could have done these u,v to d,e before the duplication
            __m128i e16__0_7 = _mm_slli_epi16(_mm_subs_epi16(v16__0_7, _mm_set1_epi16(128)), 4);
            __m128i r16__0_7 = _mm_min_epi16(_mm_set1_epi16(255), _mm_max_epi16(zero, ((_mm_add_epi16(_mm_mulhi_epi16(c16__0_7, n298), _mm_mulhi_epi16(e16__0_7, n409))))));                                                 // (298 * c + 409 * e + 128) ; //
            __m128i g16__0_7 = _mm_min_epi16(_mm_set1_epi16(255), _mm_max_epi16(zero, ((_mm_sub_epi16(_mm_sub_epi16(_mm_mulhi_epi16(c16__0_7, n298), _mm_mulhi_epi16(d16__0_7, n100)), _mm_mulhi_epi16(e16__0_7, n208)))))); // (298 * c - 100 * d - 208 * e + 128)
            __m128i b16__0_7 = _mm_min_epi16(_mm_set1_epi16(255), _mm_max_epi16(zero, ((_mm_add_epi16(_mm_mulhi_epi16(c16__0_7, n298), _mm_mulhi_epi16(d16__0_7, n516))))));                                                 // clampbyte((298 * c + 516 * d + 128) >> 8);

            // Compute R, G, B values for second 8 pixels
            __m128i c16__8_F = _mm_slli_epi16(_mm_subs_epi16(y16__8_F, _mm_set1_epi16(16)), 4);
            __m128i d16__8_F = _mm_slli_epi16(_mm_subs_epi16(u16__8_F, _mm_set1_epi16(128)), 4); // perhaps could have done these u,v to d,e before the duplication
            __m128i e16__8_F = _mm_slli_epi16(_mm_subs_epi16(v16__8_F, _mm_set1_epi16(128)), 4);           
            __m128i r16__8_F = _mm_min_epi16(_mm_set1_epi16(255), _mm_max_epi16(zero, ((_mm_add_epi16(_mm_mulhi_epi16(c16__8_F, n298), _mm_mulhi_epi16(e16__8_F, n409))))));                                                 // (298 * c + 409 * e + 128) ; //
            __m128i g16__8_F = _mm_min_epi16(_mm_set1_epi16(255), _mm_max_epi16(zero, ((_mm_sub_epi16(_mm_sub_epi16(_mm_mulhi_epi16(c16__8_F, n298), _mm_mulhi_epi16(d16__8_F, n100)), _mm_mulhi_epi16(e16__8_F, n208)))))); // (298 * c - 100 * d - 208 * e + 128)
            __m128i b16__8_F = _mm_min_epi16(_mm_set1_epi16(255), _mm_max_epi16(zero, ((_mm_add_epi16(_mm_mulhi_epi16(c16__8_F, n298), _mm_mulhi_epi16(d16__8_F, n516))))));                                                 // clampbyte((298 * c + 516 * d + 128) >> 8);

            if (FORMAT == RS_FORMAT_RGB8 || FORMAT == RS_FORMAT_RGBA8)
            {
                // Shuffle separate R, G, B values into four registers storing four pixels each in (R, G, B, A) order
                __m128i rg8__0_7 = _mm_unpacklo_epi8(_mm_shuffle_epi8(r16__0_7, evens_odds), _mm_shuffle_epi8(g16__0_7, evens_odds)); // hi to take the odds which are the upper bytes we care about
                __m128i ba8__0_7 = _mm_unpacklo_epi8(_mm_shuffle_epi8(b16__0_7, evens_odds), _mm_set1_epi8(-1));
                __m128i rgba_0_3 = _mm_unpacklo_epi16(rg8__0_7, ba8__0_7);
                __m128i rgba_4_7 = _mm_unpackhi_epi16(rg8__0_7, ba8__0_7);

                __m128i rg8__8_F = _mm_unpacklo_epi8(_mm_shuffle_epi8(r16__8_F, evens_odds), _mm_shuffle_epi8(g16__8_F, evens_odds)); // hi to take the odds which are the upper bytes we care about
                __m128i ba8__8_F = _mm_unpacklo_epi8(_mm_shuffle_epi8(b16__8_F, evens_odds), _mm_set1_epi8(-1));
                __m128i rgba_8_B = _mm_unpacklo_epi16(rg8__8_F, ba8__8_F);
                __m128i rgba_C_F = _mm_unpackhi_epi16(rg8__8_F, ba8__8_F);

                if(FORMAT == RS_FORMAT_RGBA8)
                {
                    // Store 16 pixels (64 bytes) at once
                    _mm_storeu_si128(dst++, rgba_0_3);
                    _mm_storeu_si128(dst++, rgba_4_7);
                    _mm_storeu_si128(dst++, rgba_8_B);
                    _mm_storeu_si128(dst++, rgba_C_F);
                }

                if(FORMAT == RS_FORMAT_RGB8)
                {
                    // Shuffle rgb triples to the start and end of each register
                    __m128i rgb0 = _mm_shuffle_epi8(rgba_0_3, _mm_setr_epi8(  3, 7, 11, 15,   0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14));
                    __m128i rgb1 = _mm_shuffle_epi8(rgba_4_7, _mm_setr_epi8(0, 1, 2, 4,   3, 7, 11, 15,   5, 6, 8, 9, 10, 12, 13, 14));
                    __m128i rgb2 = _mm_shuffle_epi8(rgba_8_B, _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9,   3, 7, 11, 15,   10, 12, 13, 14));
                    __m128i rgb3 = _mm_shuffle_epi8(rgba_C_F, _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14,   3, 7, 11, 15  ));

                    // Align registers and store 16 pixels (48 bytes) at once
                    _mm_storeu_si128(dst++, _mm_alignr_epi8(rgb1, rgb0, 4));
                    _mm_storeu_si128(dst++, _mm_alignr_epi8(rgb2, rgb1, 8));
                    _mm_storeu_si128(dst++, _mm_alignr_epi8(rgb3, rgb2, 12));                
                }
            }

            if (FORMAT == RS_FORMAT_BGR8 || FORMAT == RS_FORMAT_BGRA8)
            {
                // Shuffle separate R, G, B values into four registers storing four pixels each in (B, G, R, A) order
                __m128i bg8__0_7 = _mm_unpacklo_epi8(_mm_shuffle_epi8(b16__0_7, evens_odds), _mm_shuffle_epi8(g16__0_7, evens_odds)); // hi to take the odds which are the upper bytes we care about
                __m128i ra8__0_7 = _mm_unpacklo_epi8(_mm_shuffle_epi8(r16__0_7, evens_odds), _mm_set1_epi8(-1));
                __m128i bgra_0_3 = _mm_unpacklo_epi16(bg8__0_7, ra8__0_7);
                __m128i bgra_4_7 = _mm_unpackhi_epi16(bg8__0_7, ra8__0_7);

                __m128i bg8__8_F = _mm_unpacklo_epi8(_mm_shuffle_epi8(b16__8_F, evens_odds), _mm_shuffle_epi8(g16__8_F, evens_odds)); // hi to take the odds which are the upper bytes we care about
                __m128i ra8__8_F = _mm_unpacklo_epi8(_mm_shuffle_epi8(r16__8_F, evens_odds), _mm_set1_epi8(-1));
                __m128i bgra_8_B = _mm_unpacklo_epi16(bg8__8_F, ra8__8_F);
                __m128i bgra_C_F = _mm_unpackhi_epi16(bg8__8_F, ra8__8_F);

                if(FORMAT == RS_FORMAT_BGRA8)
                {
                    // Store 16 pixels (64 bytes) at once
                    _mm_storeu_si128(dst++, bgra_0_3);
                    _mm_storeu_si128(dst++, bgra_4_7);
                    _mm_storeu_si128(dst++, bgra_8_B);
                    _mm_storeu_si128(dst++, bgra_C_F);
                }

                if(FORMAT == RS_FORMAT_BGR8)
                {
                    // Shuffle rgb triples to the start and end of each register
                    __m128i bgr0 = _mm_shuffle_epi8(bgra_0_3, _mm_setr_epi8(  3, 7, 11, 15,   0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14));
                    __m128i bgr1 = _mm_shuffle_epi8(bgra_4_7, _mm_setr_epi8(0, 1, 2, 4,   3, 7, 11, 15,   5, 6, 8, 9, 10, 12, 13, 14));
                    __m128i bgr2 = _mm_shuffle_epi8(bgra_8_B, _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9,   3, 7, 11, 15,   10, 12, 13, 14));
                    __m128i bgr3 = _mm_shuffle_epi8(bgra_C_F, _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14,   3, 7, 11, 15  ));

                    // Align registers and store 16 pixels (48 bytes) at once
                    _mm_storeu_si128(dst++, _mm_alignr_epi8(bgr1, bgr0, 4));
                    _mm_storeu_si128(dst++, _mm_alignr_epi8(bgr2, bgr1, 8));
                    _mm_storeu_si128(dst++, _mm_alignr_epi8(bgr3, bgr2, 12));                
                }
            }
        }    
    }
    
    //////////////////////////////////////
    // 2-in-1 format splitting routines //
    //////////////////////////////////////

    template<class SOURCE, class SPLIT_A, class SPLIT_B> void split_frame(byte * const dest[], int count, const SOURCE * source, SPLIT_A split_a, SPLIT_B split_b)
    {
        auto a = reinterpret_cast<decltype(split_a(SOURCE())) *>(dest[0]);
        auto b = reinterpret_cast<decltype(split_b(SOURCE())) *>(dest[1]);
        for(int i=0; i<count; ++i)
        {
            *a++ = split_a(*source);
            *b++ = split_b(*source++);
        }    
    }

    struct y8i_pixel { uint8_t l, r; };
    void unpack_y8_y8_from_y8i(byte * const dest[], const byte * source, int count)
    {
        split_frame(dest, count, reinterpret_cast<const y8i_pixel *>(source),
            [](const y8i_pixel & p) -> uint8_t { return p.l; },
            [](const y8i_pixel & p) -> uint8_t { return p.r; });
    }

    struct y12i_pixel { uint8_t rl : 8, rh : 4, ll : 4, lh : 8; int l() const { return lh << 4 | ll; } int r() const { return rh << 8 | rl; } };
    void unpack_y16_y16_from_y12i_10(byte * const dest[], const byte * source, int count)
    {
        split_frame(dest, count, reinterpret_cast<const y12i_pixel *>(source),
            [](const y12i_pixel & p) -> uint16_t { return p.l() << 6 | p.l() >> 4; },  // We want to convert 10-bit data to 16-bit data
            [](const y12i_pixel & p) -> uint16_t { return p.r() << 6 | p.r() >> 4; }); // Multiply by 64 1/16 to efficiently approximate 65535/1023
    }

    struct f200_inzi_pixel { uint16_t z16; uint8_t y8; };
    void unpack_z16_y8_from_f200_inzi(byte * const dest[], const byte * source, int count)
    {
        split_frame(dest, count, reinterpret_cast<const f200_inzi_pixel *>(source),
            [](const f200_inzi_pixel & p) -> uint16_t { return p.z16; },
            [](const f200_inzi_pixel & p) -> uint8_t { return p.y8; });
    }

    void unpack_z16_y16_from_f200_inzi(byte * const dest[], const byte * source, int count)
    {
        split_frame(dest, count, reinterpret_cast<const f200_inzi_pixel *>(source),
            [](const f200_inzi_pixel & p) -> uint16_t { return p.z16; },
            [](const f200_inzi_pixel & p) -> uint16_t { return p.y8 | p.y8 << 8; });
    }

    void unpack_z16_y8_from_sr300_inzi(byte * const dest[], const byte * source, int count)
    {
        auto in = reinterpret_cast<const uint16_t *>(source);
        auto out_ir = reinterpret_cast<uint8_t *>(dest[1]);            
        for(int i=0; i<count; ++i) *out_ir++ = *in++ >> 2;
        memcpy(dest[0], in, count*2);
    }

    void unpack_z16_y16_from_sr300_inzi (byte * const dest[], const byte * source, int count)
    {
        auto in = reinterpret_cast<const uint16_t *>(source);
        auto out_ir = reinterpret_cast<uint16_t *>(dest[1]);
        for(int i=0; i<count; ++i) *out_ir++ = *in++ << 6;
        memcpy(dest[0], in, count*2);
    }

    //////////////////////////
    // Native pixel formats //
    //////////////////////////

    const native_pixel_format pf_rw10       = {'RW10', 1, 1, {{&copy_pixels<1>,                 {{RS_STREAM_COLOR,    RS_FORMAT_RAW10}}}}};
    const native_pixel_format pf_yuy2       = {'YUY2', 1, 2, {{&copy_pixels<2>,                 {{RS_STREAM_COLOR,    RS_FORMAT_YUYV }}},
                                                              {&unpack_yuy2<RS_FORMAT_Y8   >,   {{RS_STREAM_COLOR,    RS_FORMAT_Y8   }}},
                                                              {&unpack_yuy2<RS_FORMAT_Y16  >,   {{RS_STREAM_COLOR,    RS_FORMAT_Y16  }}},
                                                              {&unpack_yuy2<RS_FORMAT_RGB8 >,   {{RS_STREAM_COLOR,    RS_FORMAT_RGB8 }}},
                                                              {&unpack_yuy2<RS_FORMAT_RGBA8>,   {{RS_STREAM_COLOR,    RS_FORMAT_RGBA8}}},
                                                              {&unpack_yuy2<RS_FORMAT_BGR8 >,   {{RS_STREAM_COLOR,    RS_FORMAT_BGR8 }}},
                                                              {&unpack_yuy2<RS_FORMAT_BGRA8>,   {{RS_STREAM_COLOR,    RS_FORMAT_BGRA8}}}}};
    const native_pixel_format pf_y8         = {'Y8  ', 1, 1, {{&copy_pixels<1>,                 {{RS_STREAM_INFRARED, RS_FORMAT_Y8   }}}}};
    const native_pixel_format pf_y16        = {'Y16 ', 1, 2, {{&unpack_y16_from_y16_10,         {{RS_STREAM_INFRARED, RS_FORMAT_Y16  }}}}};
    const native_pixel_format pf_y8i        = {'Y8I ', 1, 2, {{&unpack_y8_y8_from_y8i,          {{RS_STREAM_INFRARED, RS_FORMAT_Y8   }, {RS_STREAM_INFRARED2, RS_FORMAT_Y8}}}}};
    const native_pixel_format pf_y12i       = {'Y12I', 1, 3, {{&unpack_y16_y16_from_y12i_10,    {{RS_STREAM_INFRARED, RS_FORMAT_Y16  }, {RS_STREAM_INFRARED2, RS_FORMAT_Y16}}}}};
    const native_pixel_format pf_z16        = {'Z16 ', 1, 2, {{&copy_pixels<2>,                 {{RS_STREAM_DEPTH,    RS_FORMAT_Z16  }}},
                                                              {&copy_pixels<2>,                 {{RS_STREAM_DEPTH,    RS_FORMAT_DISPARITY16}}}}};
    const native_pixel_format pf_invz       = {'INVZ', 1, 2, {{&copy_pixels<2>,                 {{RS_STREAM_DEPTH,    RS_FORMAT_Z16  }}}}};
    const native_pixel_format pf_f200_invi  = {'INVI', 1, 1, {{&copy_pixels<1>,                 {{RS_STREAM_INFRARED, RS_FORMAT_Y8   }}},
                                                              {&unpack_y16_from_y8,             {{RS_STREAM_INFRARED, RS_FORMAT_Y16  }}}}};
    const native_pixel_format pf_f200_inzi  = {'INZI', 1, 3, {{&unpack_z16_y8_from_f200_inzi,   {{RS_STREAM_DEPTH,    RS_FORMAT_Z16  }, {RS_STREAM_INFRARED, RS_FORMAT_Y8}}},
                                                              {&unpack_z16_y16_from_f200_inzi,  {{RS_STREAM_DEPTH,    RS_FORMAT_Z16  }, {RS_STREAM_INFRARED, RS_FORMAT_Y16}}}}};
    const native_pixel_format pf_sr300_invi = {'INVI', 1, 2, {{&unpack_y8_from_y16_10,          {{RS_STREAM_INFRARED, RS_FORMAT_Y8   }}},
                                                              {&unpack_y16_from_y16_10,         {{RS_STREAM_INFRARED, RS_FORMAT_Y16  }}}}};
    const native_pixel_format pf_sr300_inzi = {'INZI', 2, 2, {{&unpack_z16_y8_from_sr300_inzi,  {{RS_STREAM_DEPTH,    RS_FORMAT_Z16  }, {RS_STREAM_INFRARED, RS_FORMAT_Y8}}},
                                                              {&unpack_z16_y16_from_sr300_inzi, {{RS_STREAM_DEPTH,    RS_FORMAT_Z16  }, {RS_STREAM_INFRARED, RS_FORMAT_Y16}}}}};

    //////////////////
    // Deprojection //
    //////////////////

    template<class MAP_DEPTH> void deproject_depth(float * points, const rs_intrinsics & intrin, const uint16_t * depth, MAP_DEPTH map_depth)
    {
        for(int y=0; y<intrin.height; ++y)
        {
            for(int x=0; x<intrin.width; ++x)
            {
                const float pixel[] = { (float) x, (float) y};
                rs_deproject_pixel_to_point(points, &intrin, pixel, map_depth(*depth++));
                points += 3;                
            }
        }    
    }

    void deproject_z(float * points, const rs_intrinsics & z_intrin, const uint16_t * z_pixels, float z_scale)
    {
        deproject_depth(points, z_intrin, z_pixels, [z_scale](uint16_t z) { return z_scale * z; });
    }

    void deproject_disparity(float * points, const rs_intrinsics & disparity_intrin, const uint16_t * disparity_pixels, float disparity_scale)
    {
        deproject_depth(points, disparity_intrin, disparity_pixels, [disparity_scale](uint16_t disparity) { return disparity_scale / disparity; });
    }

    /////////////////////
    // Image alignment //
    /////////////////////

    template<class GET_DEPTH, class TRANSFER_PIXEL> void align_images(const rs_intrinsics & depth_intrin, const rs_extrinsics & depth_to_other, const rs_intrinsics & other_intrin, GET_DEPTH get_depth, TRANSFER_PIXEL transfer_pixel)
    {
        // Iterate over the pixels of the depth image    
        for(int depth_y = 0, depth_pixel_index = 0; depth_y < depth_intrin.height; ++depth_y)
        {
            for(int depth_x = 0; depth_x < depth_intrin.width; ++depth_x, ++depth_pixel_index)
            {
                // Skip over depth pixels with the value of zero, we have no depth data so we will not write anything into our aligned images
                if(float depth = get_depth(depth_pixel_index))
                {
                    // Map the top-left corner of the depth pixel onto the other image
                    float depth_pixel[2] = {depth_x-0.5f, depth_y-0.5f}, depth_point[3], other_point[3], other_pixel[2];
                    rs_deproject_pixel_to_point(depth_point, &depth_intrin, depth_pixel, depth);
                    rs_transform_point_to_point(other_point, &depth_to_other, depth_point);
                    rs_project_point_to_pixel(other_pixel, &other_intrin, other_point);
                    const int other_x0 = std::round(other_pixel[0]), other_y0 = std::round(other_pixel[1]);

                    // Map the bottom-right corner of the depth pixel onto the other image
                    depth_pixel[0] = depth_x+0.5f; depth_pixel[1] = depth_y+0.5f;
                    rs_deproject_pixel_to_point(depth_point, &depth_intrin, depth_pixel, depth);
                    rs_transform_point_to_point(other_point, &depth_to_other, depth_point);
                    rs_project_point_to_pixel(other_pixel, &other_intrin, other_point);
                    const int other_x1 = std::round(other_pixel[0]), other_y1 = std::round(other_pixel[1]);

                    if(other_x0 < 0 || other_y0 < 0 || other_x1 >= other_intrin.width || other_y1 >= other_intrin.height) continue;

                    // Transfer between the depth pixels and the pixels inside the rectangle on the other image
                    for(int y=other_y0; y<=other_y1; ++y) for(int x=other_x0; x<=other_x1; ++x) transfer_pixel(depth_pixel_index, y * other_intrin.width + x);
                }
            }
        }    
    }

    void align_z_to_other(byte * z_aligned_to_other, const uint16_t * z_pixels, float z_scale, const rs_intrinsics & z_intrin, const rs_extrinsics & z_to_other, const rs_intrinsics & other_intrin)
    {
        auto out_z = (uint16_t *)(z_aligned_to_other);
        align_images(z_intrin, z_to_other, other_intrin, 
            [z_pixels, z_scale](int z_pixel_index) { return z_scale * z_pixels[z_pixel_index]; },
            [out_z, z_pixels](int z_pixel_index, int other_pixel_index) { out_z[other_pixel_index] = z_pixels[z_pixel_index]; });
    }

    void align_disparity_to_other(byte * disparity_aligned_to_other, const uint16_t * disparity_pixels, float disparity_scale, const rs_intrinsics & disparity_intrin, const rs_extrinsics & disparity_to_other, const rs_intrinsics & other_intrin)
    {
        auto out_disparity = (uint16_t *)(disparity_aligned_to_other);
        align_images(disparity_intrin, disparity_to_other, other_intrin, 
            [disparity_pixels, disparity_scale](int disparity_pixel_index) { return disparity_scale / disparity_pixels[disparity_pixel_index]; },
            [out_disparity, disparity_pixels](int disparity_pixel_index, int other_pixel_index) { out_disparity[other_pixel_index] = disparity_pixels[disparity_pixel_index]; });
    }

    template<int N> struct bytes { char b[N]; };
    template<int N, class GET_DEPTH> void align_other_to_depth_bytes(byte * other_aligned_to_depth, GET_DEPTH get_depth, const rs_intrinsics & depth_intrin, const rs_extrinsics & depth_to_other, const rs_intrinsics & other_intrin, const byte * other_pixels)
    {
        auto in_other = (const bytes<N> *)(other_pixels);
        auto out_other = (bytes<N> *)(other_aligned_to_depth);
        align_images(depth_intrin, depth_to_other, other_intrin, get_depth,
            [out_other, in_other](int depth_pixel_index, int other_pixel_index) { out_other[depth_pixel_index] = in_other[other_pixel_index]; });
    }

    template<class GET_DEPTH> void align_other_to_depth(byte * other_aligned_to_depth, GET_DEPTH get_depth, const rs_intrinsics & depth_intrin, const rs_extrinsics & depth_to_other, const rs_intrinsics & other_intrin, const byte * other_pixels, rs_format other_format)
    {
        switch(other_format)
        {
        case RS_FORMAT_Y8: 
            return align_other_to_depth_bytes<1>(other_aligned_to_depth, get_depth, depth_intrin, depth_to_other, other_intrin, other_pixels);
        case RS_FORMAT_Y16: case RS_FORMAT_Z16: 
            return align_other_to_depth_bytes<2>(other_aligned_to_depth, get_depth, depth_intrin, depth_to_other, other_intrin, other_pixels);
        case RS_FORMAT_RGB8: case RS_FORMAT_BGR8: 
            return align_other_to_depth_bytes<3>(other_aligned_to_depth, get_depth, depth_intrin, depth_to_other, other_intrin, other_pixels);
        case RS_FORMAT_RGBA8: case RS_FORMAT_BGRA8: 
            return align_other_to_depth_bytes<4>(other_aligned_to_depth, get_depth, depth_intrin, depth_to_other, other_intrin, other_pixels);
        default: 
            assert(false); // NOTE: rs_align_other_to_depth_bytes<2>(...) is not appropriate for RS_FORMAT_YUYV/RS_FORMAT_RAW10 images, no logic prevents U/V channels from being written to one another
        }
    }

    void align_other_to_z(byte * other_aligned_to_z, const uint16_t * z_pixels, float z_scale, const rs_intrinsics & z_intrin, const rs_extrinsics & z_to_other, const rs_intrinsics & other_intrin, const byte * other_pixels, rs_format other_format)
    {
        align_other_to_depth(other_aligned_to_z, [z_pixels, z_scale](int z_pixel_index) { return z_scale * z_pixels[z_pixel_index]; }, z_intrin, z_to_other, other_intrin, other_pixels, other_format);
    }

    void align_other_to_disparity(byte * other_aligned_to_disparity, const uint16_t * disparity_pixels, float disparity_scale, const rs_intrinsics & disparity_intrin, const rs_extrinsics & disparity_to_other, const rs_intrinsics & other_intrin, const byte * other_pixels, rs_format other_format)
    {
        align_other_to_depth(other_aligned_to_disparity, [disparity_pixels, disparity_scale](int disparity_pixel_index) { return disparity_scale / disparity_pixels[disparity_pixel_index]; }, disparity_intrin, disparity_to_other, other_intrin, other_pixels, other_format);
    }

    /////////////////////////
    // Image rectification //
    /////////////////////////

    std::vector<int> compute_rectification_table(const rs_intrinsics & rect_intrin, const rs_extrinsics & rect_to_unrect, const rs_intrinsics & unrect_intrin)
    {   
        std::vector<int> rectification_table;
        rectification_table.resize(rect_intrin.width * rect_intrin.height);
        align_images(rect_intrin, rect_to_unrect, unrect_intrin, [](int) { return 1.0f; },
            [&rectification_table](int rect_pixel_index, int unrect_pixel_index) { rectification_table[rect_pixel_index] = unrect_pixel_index; });
        return rectification_table;
    }

    template<class T> void rectify_image_pixels(T * rect_pixels, const std::vector<int> & rectification_table, const T * unrect_pixels)
    {
        for(auto entry : rectification_table) *rect_pixels++ = unrect_pixels[entry];
    }

    void rectify_image(byte * rect_pixels, const std::vector<int> & rectification_table, const byte * unrect_pixels, rs_format format)
    {
        switch(format)
        {
        case RS_FORMAT_Y8: 
            return rectify_image_pixels((bytes<1> *)rect_pixels, rectification_table, (const bytes<1> *)unrect_pixels);
        case RS_FORMAT_Y16: case RS_FORMAT_Z16: 
            return rectify_image_pixels((bytes<2> *)rect_pixels, rectification_table, (const bytes<2> *)unrect_pixels);
        case RS_FORMAT_RGB8: case RS_FORMAT_BGR8: 
            return rectify_image_pixels((bytes<3> *)rect_pixels, rectification_table, (const bytes<3> *)unrect_pixels);
        case RS_FORMAT_RGBA8: case RS_FORMAT_BGRA8: 
            return rectify_image_pixels((bytes<4> *)rect_pixels, rectification_table, (const bytes<4> *)unrect_pixels);
        default: 
            assert(false); // NOTE: rectify_image_pixels(...) is not appropriate for RS_FORMAT_YUYV images, no logic prevents U/V channels from being written to one another
        }
    }
}

#pragma pack(pop)
