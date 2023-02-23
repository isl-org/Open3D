// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/core/Dtype.h"
#include "open3d/utility/Logging.h"

/// Call a numerical templated function based on Dtype. Wrap the function to
/// a lambda function to use DISPATCH_DTYPE_TO_TEMPLATE.
///
/// Before:
///     if (dtype == core::Float32) {
///         func<float>(args);
///     } else if (dtype == core::Float64) {
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
#define DISPATCH_DTYPE_TO_TEMPLATE(DTYPE, ...)                   \
    [&] {                                                        \
        if (DTYPE == open3d::core::Float32) {                    \
            using scalar_t = float;                              \
            return __VA_ARGS__();                                \
        } else if (DTYPE == open3d::core::Float64) {             \
            using scalar_t = double;                             \
            return __VA_ARGS__();                                \
        } else if (DTYPE == open3d::core::Int8) {                \
            using scalar_t = int8_t;                             \
            return __VA_ARGS__();                                \
        } else if (DTYPE == open3d::core::Int16) {               \
            using scalar_t = int16_t;                            \
            return __VA_ARGS__();                                \
        } else if (DTYPE == open3d::core::Int32) {               \
            using scalar_t = int32_t;                            \
            return __VA_ARGS__();                                \
        } else if (DTYPE == open3d::core::Int64) {               \
            using scalar_t = int64_t;                            \
            return __VA_ARGS__();                                \
        } else if (DTYPE == open3d::core::UInt8) {               \
            using scalar_t = uint8_t;                            \
            return __VA_ARGS__();                                \
        } else if (DTYPE == open3d::core::UInt16) {              \
            using scalar_t = uint16_t;                           \
            return __VA_ARGS__();                                \
        } else if (DTYPE == open3d::core::UInt32) {              \
            using scalar_t = uint32_t;                           \
            return __VA_ARGS__();                                \
        } else if (DTYPE == open3d::core::UInt64) {              \
            using scalar_t = uint64_t;                           \
            return __VA_ARGS__();                                \
        } else {                                                 \
            open3d::utility::LogError("Unsupported data type."); \
        }                                                        \
    }()

#define DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(DTYPE, ...)    \
    [&] {                                                   \
        if (DTYPE == open3d::core::Bool) {                  \
            using scalar_t = bool;                          \
            return __VA_ARGS__();                           \
        } else {                                            \
            DISPATCH_DTYPE_TO_TEMPLATE(DTYPE, __VA_ARGS__); \
        }                                                   \
    }()

#define DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(DTYPE, ...)             \
    [&] {                                                        \
        if (DTYPE == open3d::core::Float32) {                    \
            using scalar_t = float;                              \
            return __VA_ARGS__();                                \
        } else if (DTYPE == open3d::core::Float64) {             \
            using scalar_t = double;                             \
            return __VA_ARGS__();                                \
        } else {                                                 \
            open3d::utility::LogError("Unsupported data type."); \
        }                                                        \
    }()

#define DISPATCH_FLOAT_INT_DTYPE_TO_TEMPLATE(FDTYPE, IDTYPE, ...) \
    [&] {                                                         \
        if (FDTYPE == open3d::core::Float32 &&                    \
            IDTYPE == open3d::core::Int32) {                      \
            using scalar_t = float;                               \
            using int_t = int32_t;                                \
            return __VA_ARGS__();                                 \
        } else if (FDTYPE == open3d::core::Float32 &&             \
                   IDTYPE == open3d::core::Int64) {               \
            using scalar_t = float;                               \
            using int_t = int64_t;                                \
            return __VA_ARGS__();                                 \
        } else if (FDTYPE == open3d::core::Float64 &&             \
                   IDTYPE == open3d::core::Int32) {               \
            using scalar_t = double;                              \
            using int_t = int32_t;                                \
            return __VA_ARGS__();                                 \
        } else if (FDTYPE == open3d::core::Float64 &&             \
                   IDTYPE == open3d::core::Int64) {               \
            using scalar_t = double;                              \
            using int_t = int64_t;                                \
            return __VA_ARGS__();                                 \
        } else {                                                  \
            open3d::utility::LogError("Unsupported data type.");  \
        }                                                         \
    }()
