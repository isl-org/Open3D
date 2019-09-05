// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#pragma once

// clang-format off
#define ARGS_1(a1, ...) a1
#define ARGS_2(a1, a2, ...) a2
#define ARGS_3(a1, a2, a3, ...) a3
#define ARGS_4(a1, a2, a3, a4, ...) a4
#define ARGS_5(a1, a2, a3, a4, a5, ...) a5
#define ARGS_6(a1, a2, a3, a4, a5, a6, ...) a6
#define ARGS_7(a1, a2, a3, a4, a5, a6, a7, ...) a7
#define ARGS_8(a1, a2, a3, a4, a5, a6, a7, a8, ...) a8
#define ARGS_9(a1, a2, a3, a4, a5, a6, a7, a8, a9, ...) a9
#define ARGS_10(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, ...) a10
#define ARGS_11(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, ...) a11
#define ARGS_12(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, ...) a12
#define ARGS_13(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, ...) a13
#define ARGS_14(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, ...) a14
#define ARGS_15(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, ...) a15
#define ARGS_16(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, ...) a16
#define ARGS_17(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, ...) a17
#define ARGS_18(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, ...) a18
#define ARGS_19(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, ...) a19
#define ARGS_20(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, ...) a20
#define ARGS_21(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, ...) a21
// clang-format on

#define EXPAND(x) x  // MSVC fix

// Count __VA_ARGS__ for MSVC https://stackoverflow.com/a/26685339/1255535
#ifdef _MSC_VER

#define __NARGS(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, \
                _15, _16, _17, _18, _19, _20, _21, VAL, ...)                 \
    VAL
#define NARGS_1(...)                                                           \
    EXPAND(__NARGS(__VA_ARGS__, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, \
                   8, 7, 6, 5, 4, 3, 2, 1, 0))
#define AUGMENTER(...) unused, __VA_ARGS__
#define COUNT_ARGS(...) NARGS_1(AUGMENTER(__VA_ARGS__))

#else
#define COUNT_ARGS(...)                                                       \
    __NARGS(0, ##__VA_ARGS__, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, \
            9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
#define __NARGS(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, \
                _14, _15, _16, _17, _18, _19, _20, _21, N, ...)             \
    N

#endif

// Convert to list of "arg_type arg_name"
// Converts:
//     float, a1, int, a2, short, a3, double, a4
// To:
//     float a1, int a2, short a3, double a4
// clang-format off
#define EXTRACT_TYPES_PARAMS_0(...)

// Workaround: with older compilers ##__VA_ARGS__ may not eliminate unnecessary
// comma when __VA_ARGS__ is empty.
// In this case, COUNT_ARGS() will be 1 where it should be 0.
#define EXTRACT_TYPES_PARAMS_1(...)

#define EXTRACT_TYPES_PARAMS_2(...) \
    EXPAND(ARGS_1(__VA_ARGS__)) EXPAND(ARGS_2(__VA_ARGS__))

#define EXTRACT_TYPES_PARAMS_4(...) \
    EXPAND(ARGS_1(__VA_ARGS__)) EXPAND(ARGS_2(__VA_ARGS__)), \
    EXPAND(ARGS_3(__VA_ARGS__)) EXPAND(ARGS_4(__VA_ARGS__))

#define EXTRACT_TYPES_PARAMS_6(...) \
    EXPAND(ARGS_1(__VA_ARGS__)) EXPAND(ARGS_2(__VA_ARGS__)), \
    EXPAND(ARGS_3(__VA_ARGS__)) EXPAND(ARGS_4(__VA_ARGS__)), \
    EXPAND(ARGS_5(__VA_ARGS__)) EXPAND(ARGS_6(__VA_ARGS__))

#define EXTRACT_TYPES_PARAMS_8(...) \
    EXPAND(ARGS_1 (__VA_ARGS__)) EXPAND(ARGS_2 (__VA_ARGS__)), \
    EXPAND(ARGS_3 (__VA_ARGS__)) EXPAND(ARGS_4 (__VA_ARGS__)), \
    EXPAND(ARGS_5 (__VA_ARGS__)) EXPAND(ARGS_6 (__VA_ARGS__)), \
    EXPAND(ARGS_7 (__VA_ARGS__)) EXPAND(ARGS_8 (__VA_ARGS__))

#define EXTRACT_TYPES_PARAMS_10(...) \
    EXPAND(ARGS_1 (__VA_ARGS__)) EXPAND(ARGS_2 (__VA_ARGS__)), \
    EXPAND(ARGS_3 (__VA_ARGS__)) EXPAND(ARGS_4 (__VA_ARGS__)), \
    EXPAND(ARGS_5 (__VA_ARGS__)) EXPAND(ARGS_6 (__VA_ARGS__)), \
    EXPAND(ARGS_7 (__VA_ARGS__)) EXPAND(ARGS_8 (__VA_ARGS__)), \
    EXPAND(ARGS_9 (__VA_ARGS__)) EXPAND(ARGS_10(__VA_ARGS__))

#define EXTRACT_TYPES_PARAMS_12(...) \
    EXPAND(ARGS_1 (__VA_ARGS__)) EXPAND(ARGS_2 (__VA_ARGS__)), \
    EXPAND(ARGS_3 (__VA_ARGS__)) EXPAND(ARGS_4 (__VA_ARGS__)), \
    EXPAND(ARGS_5 (__VA_ARGS__)) EXPAND(ARGS_6 (__VA_ARGS__)), \
    EXPAND(ARGS_7 (__VA_ARGS__)) EXPAND(ARGS_8 (__VA_ARGS__)), \
    EXPAND(ARGS_9 (__VA_ARGS__)) EXPAND(ARGS_10(__VA_ARGS__)), \
    EXPAND(ARGS_11(__VA_ARGS__)) EXPAND(ARGS_12(__VA_ARGS__))

#define EXTRACT_TYPES_PARAMS_14(...) \
    EXPAND(ARGS_1 (__VA_ARGS__)) EXPAND(ARGS_2 (__VA_ARGS__)), \
    EXPAND(ARGS_3 (__VA_ARGS__)) EXPAND(ARGS_4 (__VA_ARGS__)), \
    EXPAND(ARGS_5 (__VA_ARGS__)) EXPAND(ARGS_6 (__VA_ARGS__)), \
    EXPAND(ARGS_7 (__VA_ARGS__)) EXPAND(ARGS_8 (__VA_ARGS__)), \
    EXPAND(ARGS_9 (__VA_ARGS__)) EXPAND(ARGS_10(__VA_ARGS__)), \
    EXPAND(ARGS_11(__VA_ARGS__)) EXPAND(ARGS_12(__VA_ARGS__)), \
    EXPAND(ARGS_13(__VA_ARGS__)) EXPAND(ARGS_14(__VA_ARGS__))

#define EXTRACT_TYPES_PARAMS_16(...) \
    EXPAND(ARGS_1 (__VA_ARGS__)) EXPAND(ARGS_2 (__VA_ARGS__)), \
    EXPAND(ARGS_3 (__VA_ARGS__)) EXPAND(ARGS_4 (__VA_ARGS__)), \
    EXPAND(ARGS_5 (__VA_ARGS__)) EXPAND(ARGS_6 (__VA_ARGS__)), \
    EXPAND(ARGS_7 (__VA_ARGS__)) EXPAND(ARGS_8 (__VA_ARGS__)), \
    EXPAND(ARGS_9 (__VA_ARGS__)) EXPAND(ARGS_10(__VA_ARGS__)), \
    EXPAND(ARGS_11(__VA_ARGS__)) EXPAND(ARGS_12(__VA_ARGS__)), \
    EXPAND(ARGS_13(__VA_ARGS__)) EXPAND(ARGS_14(__VA_ARGS__)), \
    EXPAND(ARGS_15(__VA_ARGS__)) EXPAND(ARGS_16(__VA_ARGS__))

#define EXTRACT_TYPES_PARAMS_18(...) \
    EXPAND(ARGS_1 (__VA_ARGS__)) EXPAND(ARGS_2 (__VA_ARGS__)), \
    EXPAND(ARGS_3 (__VA_ARGS__)) EXPAND(ARGS_4 (__VA_ARGS__)), \
    EXPAND(ARGS_5 (__VA_ARGS__)) EXPAND(ARGS_6 (__VA_ARGS__)), \
    EXPAND(ARGS_7 (__VA_ARGS__)) EXPAND(ARGS_8 (__VA_ARGS__)), \
    EXPAND(ARGS_9 (__VA_ARGS__)) EXPAND(ARGS_10(__VA_ARGS__)), \
    EXPAND(ARGS_11(__VA_ARGS__)) EXPAND(ARGS_12(__VA_ARGS__)), \
    EXPAND(ARGS_13(__VA_ARGS__)) EXPAND(ARGS_14(__VA_ARGS__)), \
    EXPAND(ARGS_15(__VA_ARGS__)) EXPAND(ARGS_16(__VA_ARGS__)), \
    EXPAND(ARGS_17(__VA_ARGS__)) EXPAND(ARGS_18(__VA_ARGS__))

#define EXTRACT_TYPES_PARAMS_20(...) \
    EXPAND(ARGS_1 (__VA_ARGS__)) EXPAND(ARGS_2 (__VA_ARGS__)), \
    EXPAND(ARGS_3 (__VA_ARGS__)) EXPAND(ARGS_4 (__VA_ARGS__)), \
    EXPAND(ARGS_5 (__VA_ARGS__)) EXPAND(ARGS_6 (__VA_ARGS__)), \
    EXPAND(ARGS_7 (__VA_ARGS__)) EXPAND(ARGS_8 (__VA_ARGS__)), \
    EXPAND(ARGS_9 (__VA_ARGS__)) EXPAND(ARGS_10(__VA_ARGS__)), \
    EXPAND(ARGS_11(__VA_ARGS__)) EXPAND(ARGS_12(__VA_ARGS__)), \
    EXPAND(ARGS_13(__VA_ARGS__)) EXPAND(ARGS_14(__VA_ARGS__)), \
    EXPAND(ARGS_15(__VA_ARGS__)) EXPAND(ARGS_16(__VA_ARGS__)), \
    EXPAND(ARGS_17(__VA_ARGS__)) EXPAND(ARGS_18(__VA_ARGS__)), \
    EXPAND(ARGS_19(__VA_ARGS__)) EXPAND(ARGS_20(__VA_ARGS__))
// clang-format on

// Convert to list of "arg_type arg_name" caller
#define EXTRACT_TYPES_PARAMS(num_args, ...) \
    EXTRACT_TYPES_PARAMS_##num_args(__VA_ARGS__)

// Convert to list of "arg_name"
// Converts:
//     float, a1, int, a2, short, a3, double, a4
// To:
//     a1, a2, a3, a4
// clang-format off
#define EXTRACT_PARAMS_0(...)

// Workaround: with older compilers ##__VA_ARGS__ may not eliminate unnecessary
// comma when __VA_ARGS__ is empty.
// In this case, COUNT_ARGS() will be 1 where it should be 0.
#define EXTRACT_PARAMS_1(...)

#define EXTRACT_PARAMS_2(...) \
    EXPAND(ARGS_2(__VA_ARGS__))

#define EXTRACT_PARAMS_4(...) \
    EXPAND(ARGS_2(__VA_ARGS__)), EXPAND(ARGS_4(__VA_ARGS__))

#define EXTRACT_PARAMS_6(...) \
    EXPAND(ARGS_2(__VA_ARGS__)), EXPAND(ARGS_4(__VA_ARGS__)), \
    EXPAND(ARGS_6(__VA_ARGS__))

#define EXTRACT_PARAMS_8(...) \
    EXPAND(ARGS_2(__VA_ARGS__)), EXPAND(ARGS_4(__VA_ARGS__)), \
    EXPAND(ARGS_6(__VA_ARGS__)), EXPAND(ARGS_8(__VA_ARGS__))

#define EXTRACT_PARAMS_10(...) \
    EXPAND(ARGS_2 (__VA_ARGS__)), EXPAND(ARGS_4 (__VA_ARGS__)), \
    EXPAND(ARGS_6 (__VA_ARGS__)), EXPAND(ARGS_8 (__VA_ARGS__)), \
    EXPAND(ARGS_10(__VA_ARGS__))

#define EXTRACT_PARAMS_12(...) \
    EXPAND(ARGS_2 (__VA_ARGS__)), EXPAND(ARGS_4 (__VA_ARGS__)), \
    EXPAND(ARGS_6 (__VA_ARGS__)), EXPAND(ARGS_8 (__VA_ARGS__)), \
    EXPAND(ARGS_10(__VA_ARGS__)), EXPAND(ARGS_12(__VA_ARGS__))

#define EXTRACT_PARAMS_14(...) \
    EXPAND(ARGS_2 (__VA_ARGS__)), EXPAND(ARGS_4 (__VA_ARGS__)), \
    EXPAND(ARGS_6 (__VA_ARGS__)), EXPAND(ARGS_8 (__VA_ARGS__)), \
    EXPAND(ARGS_10(__VA_ARGS__)), EXPAND(ARGS_12(__VA_ARGS__)), \
    EXPAND(ARGS_14(__VA_ARGS__))

#define EXTRACT_PARAMS_16(...) \
    EXPAND(ARGS_2 (__VA_ARGS__)), EXPAND(ARGS_4 (__VA_ARGS__)), \
    EXPAND(ARGS_6 (__VA_ARGS__)), EXPAND(ARGS_8 (__VA_ARGS__)), \
    EXPAND(ARGS_10(__VA_ARGS__)), EXPAND(ARGS_12(__VA_ARGS__)), \
    EXPAND(ARGS_14(__VA_ARGS__)), EXPAND(ARGS_16(__VA_ARGS__))

#define EXTRACT_PARAMS_18(...) \
    EXPAND(ARGS_2 (__VA_ARGS__)), EXPAND(ARGS_4 (__VA_ARGS__)), \
    EXPAND(ARGS_6 (__VA_ARGS__)), EXPAND(ARGS_8 (__VA_ARGS__)), \
    EXPAND(ARGS_10(__VA_ARGS__)), EXPAND(ARGS_12(__VA_ARGS__)), \
    EXPAND(ARGS_14(__VA_ARGS__)), EXPAND(ARGS_16(__VA_ARGS__)), \
    EXPAND(ARGS_18(__VA_ARGS__))

#define EXTRACT_PARAMS_20(...) \
    EXPAND(ARGS_2 (__VA_ARGS__)), EXPAND(ARGS_4 (__VA_ARGS__)), \
    EXPAND(ARGS_6 (__VA_ARGS__)), EXPAND(ARGS_8 (__VA_ARGS__)), \
    EXPAND(ARGS_10(__VA_ARGS__)), EXPAND(ARGS_12(__VA_ARGS__)), \
    EXPAND(ARGS_14(__VA_ARGS__)), EXPAND(ARGS_16(__VA_ARGS__)), \
    EXPAND(ARGS_18(__VA_ARGS__)), EXPAND(ARGS_20(__VA_ARGS__))
// clang-format on

// Convert to list of "arg_name" caller
#define EXTRACT_PARAMS(num_args, ...) EXTRACT_PARAMS_##num_args(__VA_ARGS__)
