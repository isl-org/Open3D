// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// OPEN3D_FIX_MSVC_(...)
///
/// Internal helper function which defers the evaluation of the enclosed
/// expression.
///
/// Use this macro only to workaround non-compliant behaviour of the MSVC
/// preprocessor.
///
/// Note: Could be dropped in the future if the compile flag /Zc:preprocessor
/// can be applied.
#define OPEN3D_FIX_MSVC_(...) __VA_ARGS__

/// OPEN3D_CONCAT(s1, s2)
///
/// Concatenates the expanded expressions s1 and s2.
#define OPEN3D_CONCAT_IMPL_(s1, s2) s1##s2
#define OPEN3D_CONCAT(s1, s2) OPEN3D_CONCAT_IMPL_(s1, s2)

/// OPEN3D_STRINGIFY(s)
///
/// Converts the expanded expression s to a string.
#define OPEN3D_STRINGIFY_IMPL_(s) #s
#define OPEN3D_STRINGIFY(s) OPEN3D_STRINGIFY_IMPL_(s)

/// OPEN3D_NUM_ARGS(...)
///
/// Returns the number of supplied arguments.
///
/// Note: Only works for 1-10 arguments.
#define OPEN3D_GET_NTH_ARG_(...) \
    OPEN3D_FIX_MSVC_(OPEN3D_GET_NTH_ARG_IMPL_(__VA_ARGS__))
#define OPEN3D_GET_NTH_ARG_IMPL_(arg1, arg2, arg3, arg4, arg5, arg6, arg7, \
                                 arg8, arg9, arg10, N, ...)                \
    N
#define OPEN3D_REVERSE_NUM_SEQUENCE_() 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
#define OPEN3D_NUM_ARGS(...) \
    OPEN3D_GET_NTH_ARG_(__VA_ARGS__, OPEN3D_REVERSE_NUM_SEQUENCE_())

/// OPEN3D_OVERLOAD(func, ...)
///
/// Overloads the enumerated macros func1, func2, etc. based on the number of
/// additional arguments.
///
/// Example:
///
/// \code
/// #define FOO_1(x1) foo(x1)
/// #define FOO_2(x1, x2) bar(x1, x2)
/// #define FOO(...) '\'
///     OPEN3D_FIX_MSVC_(OPEN3D_OVERLOAD(FOO_, __VA_ARGS__)(__VA_ARGS__))
///
/// FOO(1)    -> foo(1)
/// FOO(2, 3) -> bar(2, 3)
/// \endcode
#define OPEN3D_OVERLOAD(func, ...) \
    OPEN3D_CONCAT(func, OPEN3D_NUM_ARGS(__VA_ARGS__))
