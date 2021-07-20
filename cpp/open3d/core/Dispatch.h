// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "open3d/core/Dtype.h"
#include "open3d/utility/Logging.h"

/// Call a numerical templated function based on Dtype. Warp the function to
/// a lambda function to use DISPATCH_DTYPE_TO_TEMPLATE.
///
/// Before:
///     if (dtype == core::kFloat32) {
///         func<float>(args);
///     } else if (dtype == core::kFloat64) {
///         func<double>(args);
///     } else ...
///
/// Now:
///     DISPATCH_DTYPE_TO_TEMPLATE(dtype, [&]() {
///        func<scalar_t>(args);
///     });
///
/// Inspired by:
///     https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h
#define DISPATCH_DTYPE_TO_TEMPLATE(DTYPE, ...)           \
    [&] {                                                \
        if (DTYPE == open3d::core::kFloat32) {           \
            using scalar_t = float;                      \
            return __VA_ARGS__();                        \
        } else if (DTYPE == open3d::core::kFloat64) {    \
            using scalar_t = double;                     \
            return __VA_ARGS__();                        \
        } else if (DTYPE == open3d::core::kInt8) {       \
            using scalar_t = int8_t;                     \
            return __VA_ARGS__();                        \
        } else if (DTYPE == open3d::core::kInt16) {      \
            using scalar_t = int16_t;                    \
            return __VA_ARGS__();                        \
        } else if (DTYPE == open3d::core::kInt32) {      \
            using scalar_t = int32_t;                    \
            return __VA_ARGS__();                        \
        } else if (DTYPE == open3d::core::kInt64) {      \
            using scalar_t = int64_t;                    \
            return __VA_ARGS__();                        \
        } else if (DTYPE == open3d::core::kUInt8) {      \
            using scalar_t = uint8_t;                    \
            return __VA_ARGS__();                        \
        } else if (DTYPE == open3d::core::kUInt16) {     \
            using scalar_t = uint16_t;                   \
            return __VA_ARGS__();                        \
        } else if (DTYPE == open3d::core::kUInt32) {     \
            using scalar_t = uint32_t;                   \
            return __VA_ARGS__();                        \
        } else if (DTYPE == open3d::core::kUInt64) {     \
            using scalar_t = uint64_t;                   \
            return __VA_ARGS__();                        \
        } else {                                         \
            utility::LogError("Unsupported data type."); \
        }                                                \
    }()

#define DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(DTYPE, ...)    \
    [&] {                                                   \
        if (DTYPE == open3d::core::kBool) {                 \
            using scalar_t = bool;                          \
            return __VA_ARGS__();                           \
        } else {                                            \
            DISPATCH_DTYPE_TO_TEMPLATE(DTYPE, __VA_ARGS__); \
        }                                                   \
    }()

#define DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(DTYPE, ...)     \
    [&] {                                                \
        if (DTYPE == open3d::core::kFloat32) {           \
            using scalar_t = float;                      \
            return __VA_ARGS__();                        \
        } else if (DTYPE == open3d::core::kFloat64) {    \
            using scalar_t = double;                     \
            return __VA_ARGS__();                        \
        } else {                                         \
            utility::LogError("Unsupported data type."); \
        }                                                \
    }()
