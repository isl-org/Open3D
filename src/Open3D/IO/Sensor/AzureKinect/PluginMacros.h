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

#define COUNT_ARGS(...)                                                      \
    ARGS_21(dummy, ##__VA_ARGS__, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, \
            8, 7, 6, 5, 4, 3, 2, 1, 0)

// Convert to list of "arg_type arg_name"
// Converts:
//     float, a1, int, a2, short, a3, double, a4
// To:
//     float a1, int a2, short a3, double a4

// clang-format off
#define EXTRACT_TYPES_PARAMS_0(...)

#define EXTRACT_TYPES_PARAMS_2(...) \
    ARGS_1(__VA_ARGS__) ARGS_2(__VA_ARGS__)

#define EXTRACT_TYPES_PARAMS_4(...) \
    ARGS_1(__VA_ARGS__) ARGS_2(__VA_ARGS__), \
    ARGS_3(__VA_ARGS__) ARGS_4(__VA_ARGS__)

#define EXTRACT_TYPES_PARAMS_6(...) \
    ARGS_1(__VA_ARGS__) ARGS_2(__VA_ARGS__), \
    ARGS_3(__VA_ARGS__) ARGS_4(__VA_ARGS__), \
    ARGS_5(__VA_ARGS__) ARGS_6(__VA_ARGS__)

#define EXTRACT_TYPES_PARAMS_8(...) \
    ARGS_1 (__VA_ARGS__) ARGS_2 (__VA_ARGS__), \
    ARGS_3 (__VA_ARGS__) ARGS_4 (__VA_ARGS__), \
    ARGS_5 (__VA_ARGS__) ARGS_6 (__VA_ARGS__), \
    ARGS_7 (__VA_ARGS__) ARGS_8 (__VA_ARGS__)

#define EXTRACT_TYPES_PARAMS_10(...) \
    ARGS_1 (__VA_ARGS__) ARGS_2 (__VA_ARGS__), \
    ARGS_3 (__VA_ARGS__) ARGS_4 (__VA_ARGS__), \
    ARGS_5 (__VA_ARGS__) ARGS_6 (__VA_ARGS__), \
    ARGS_7 (__VA_ARGS__) ARGS_8 (__VA_ARGS__), \
    ARGS_9 (__VA_ARGS__) ARGS_10(__VA_ARGS__)

#define EXTRACT_TYPES_PARAMS_12(...) \
    ARGS_1 (__VA_ARGS__) ARGS_2 (__VA_ARGS__), \
    ARGS_3 (__VA_ARGS__) ARGS_4 (__VA_ARGS__), \
    ARGS_5 (__VA_ARGS__) ARGS_6 (__VA_ARGS__), \
    ARGS_7 (__VA_ARGS__) ARGS_8 (__VA_ARGS__), \
    ARGS_9 (__VA_ARGS__) ARGS_10(__VA_ARGS__), \
    ARGS_11(__VA_ARGS__) ARGS_12(__VA_ARGS__)

#define EXTRACT_TYPES_PARAMS_14(...) \
    ARGS_1 (__VA_ARGS__) ARGS_2 (__VA_ARGS__), \
    ARGS_3 (__VA_ARGS__) ARGS_4 (__VA_ARGS__), \
    ARGS_5 (__VA_ARGS__) ARGS_6 (__VA_ARGS__), \
    ARGS_7 (__VA_ARGS__) ARGS_8 (__VA_ARGS__), \
    ARGS_9 (__VA_ARGS__) ARGS_10(__VA_ARGS__), \
    ARGS_11(__VA_ARGS__) ARGS_12(__VA_ARGS__), \
    ARGS_13(__VA_ARGS__) ARGS_14(__VA_ARGS__)

#define EXTRACT_TYPES_PARAMS_16(...) \
    ARGS_1 (__VA_ARGS__) ARGS_2 (__VA_ARGS__), \
    ARGS_3 (__VA_ARGS__) ARGS_4 (__VA_ARGS__), \
    ARGS_5 (__VA_ARGS__) ARGS_6 (__VA_ARGS__), \
    ARGS_7 (__VA_ARGS__) ARGS_8 (__VA_ARGS__), \
    ARGS_9 (__VA_ARGS__) ARGS_10(__VA_ARGS__), \
    ARGS_11(__VA_ARGS__) ARGS_12(__VA_ARGS__), \
    ARGS_13(__VA_ARGS__) ARGS_14(__VA_ARGS__), \
    ARGS_15(__VA_ARGS__) ARGS_16(__VA_ARGS__)

#define EXTRACT_TYPES_PARAMS_18(...) \
    ARGS_1 (__VA_ARGS__) ARGS_2 (__VA_ARGS__), \
    ARGS_3 (__VA_ARGS__) ARGS_4 (__VA_ARGS__), \
    ARGS_5 (__VA_ARGS__) ARGS_6 (__VA_ARGS__), \
    ARGS_7 (__VA_ARGS__) ARGS_8 (__VA_ARGS__), \
    ARGS_9 (__VA_ARGS__) ARGS_10(__VA_ARGS__), \
    ARGS_11(__VA_ARGS__) ARGS_12(__VA_ARGS__), \
    ARGS_13(__VA_ARGS__) ARGS_14(__VA_ARGS__), \
    ARGS_15(__VA_ARGS__) ARGS_16(__VA_ARGS__), \
    ARGS_17(__VA_ARGS__) ARGS_18(__VA_ARGS__)

#define EXTRACT_TYPES_PARAMS_20(...) \
    ARGS_1 (__VA_ARGS__) ARGS_2 (__VA_ARGS__), \
    ARGS_3 (__VA_ARGS__) ARGS_4 (__VA_ARGS__), \
    ARGS_5 (__VA_ARGS__) ARGS_6 (__VA_ARGS__), \
    ARGS_7 (__VA_ARGS__) ARGS_8 (__VA_ARGS__), \
    ARGS_9 (__VA_ARGS__) ARGS_10(__VA_ARGS__), \
    ARGS_11(__VA_ARGS__) ARGS_12(__VA_ARGS__), \
    ARGS_13(__VA_ARGS__) ARGS_14(__VA_ARGS__), \
    ARGS_15(__VA_ARGS__) ARGS_16(__VA_ARGS__), \
    ARGS_17(__VA_ARGS__) ARGS_18(__VA_ARGS__), \
    ARGS_19(__VA_ARGS__) ARGS_20(__VA_ARGS__)
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

#define EXTRACT_PARAMS_2(...) \
    ARGS_2(__VA_ARGS__)

#define EXTRACT_PARAMS_4(...) \
    ARGS_2(__VA_ARGS__), ARGS_4(__VA_ARGS__)

#define EXTRACT_PARAMS_6(...) \
    ARGS_2(__VA_ARGS__), ARGS_4(__VA_ARGS__), \
    ARGS_6(__VA_ARGS__)

#define EXTRACT_PARAMS_8(...) \
    ARGS_2(__VA_ARGS__), ARGS_4(__VA_ARGS__), \
    ARGS_6(__VA_ARGS__), ARGS_8(__VA_ARGS__)

#define EXTRACT_PARAMS_10(...) \
    ARGS_2 (__VA_ARGS__), ARGS_4 (__VA_ARGS__), \
    ARGS_6 (__VA_ARGS__), ARGS_8 (__VA_ARGS__), \
    ARGS_10(__VA_ARGS__)

#define EXTRACT_PARAMS_12(...) \
    ARGS_2 (__VA_ARGS__), ARGS_4 (__VA_ARGS__), \
    ARGS_6 (__VA_ARGS__), ARGS_8 (__VA_ARGS__), \
    ARGS_10(__VA_ARGS__), ARGS_12(__VA_ARGS__)

#define EXTRACT_PARAMS_14(...) \
    ARGS_2 (__VA_ARGS__), ARGS_4 (__VA_ARGS__), \
    ARGS_6 (__VA_ARGS__), ARGS_8 (__VA_ARGS__), \
    ARGS_10(__VA_ARGS__), ARGS_12(__VA_ARGS__), \
    ARGS_14(__VA_ARGS__)

#define EXTRACT_PARAMS_16(...) \
    ARGS_2 (__VA_ARGS__), ARGS_4 (__VA_ARGS__), \
    ARGS_6 (__VA_ARGS__), ARGS_8 (__VA_ARGS__), \
    ARGS_10(__VA_ARGS__), ARGS_12(__VA_ARGS__), \
    ARGS_14(__VA_ARGS__), ARGS_16(__VA_ARGS__)

#define EXTRACT_PARAMS_18(...) \
    ARGS_2 (__VA_ARGS__), ARGS_4 (__VA_ARGS__), \
    ARGS_6 (__VA_ARGS__), ARGS_8 (__VA_ARGS__), \
    ARGS_10(__VA_ARGS__), ARGS_12(__VA_ARGS__), \
    ARGS_14(__VA_ARGS__), ARGS_16(__VA_ARGS__), \
    ARGS_18(__VA_ARGS__)

#define EXTRACT_PARAMS_20(...) \
    ARGS_2 (__VA_ARGS__), ARGS_4 (__VA_ARGS__), \
    ARGS_6 (__VA_ARGS__), ARGS_8 (__VA_ARGS__), \
    ARGS_10(__VA_ARGS__), ARGS_12(__VA_ARGS__), \
    ARGS_14(__VA_ARGS__), ARGS_16(__VA_ARGS__), \
    ARGS_18(__VA_ARGS__), ARGS_20(__VA_ARGS__)
// clang-format on

// Convert to list of "arg_name" caller
#define EXTRACT_PARAMS(num_args, ...) EXTRACT_PARAMS_##num_args(__VA_ARGS__)
