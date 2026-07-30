// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// https://gcc.gnu.org/wiki/Visibility updated to use C++11 attribute syntax
// In Open3D, we set symbol visibility based on folder / cmake target through
// cmake. e.g. all symbols in kernel folders are hidden. These macros allow fine
// grained control over symbol visibility.
#if defined(_WIN32) || defined(__CYGWIN__)
#define OPEN3D_DLL_IMPORT __declspec(dllimport)
#define OPEN3D_DLL_EXPORT __declspec(dllexport)
#define OPEN3D_DLL_LOCAL
#else
#define OPEN3D_DLL_IMPORT [[gnu::visibility("default")]]
#define OPEN3D_DLL_EXPORT [[gnu::visibility("default")]]
#define OPEN3D_DLL_LOCAL [[gnu::visibility("hidden")]]
#endif

#ifdef OPEN3D_STATIC
#define OPEN3D_API
#define OPEN3D_LOCAL
#else
#define OPEN3D_LOCAL OPEN3D_DLL_LOCAL
#if defined(OPEN3D_ENABLE_DLL_EXPORTS)
#define OPEN3D_API OPEN3D_DLL_EXPORT
#else
#define OPEN3D_API OPEN3D_DLL_IMPORT
#endif
#endif

// Compiler-specific function macro.
// Ref: https://stackoverflow.com/a/4384825
#ifdef _WIN32
#define OPEN3D_FUNCTION __FUNCSIG__
#else
#define OPEN3D_FUNCTION __PRETTY_FUNCTION__
#endif

// Assertion for host, CUDA device, and SYCL device code.
// Usage:
//     OPEN3D_ASSERT(condition);
//     OPEN3D_ASSERT(condition, "Error message");
//
// CUDA and SYCL device: OPEN3D_ASSERT prints the message and traps
// unconditionally (independent of NDEBUG), so it fires in both Debug and
// Release.
// For host-only code, utility::LogError() may also be used directly.
#define OPEN3D_ASSERT_EXPAND(...) __VA_ARGS__
#define OPEN3D_ASSERT_GET_MACRO(_1, _2, NAME, ...) NAME
#define OPEN3D_ASSERT_CHOOSER(...)                \
    OPEN3D_ASSERT_EXPAND(OPEN3D_ASSERT_GET_MACRO( \
            __VA_ARGS__, OPEN3D_ASSERT_MSG, OPEN3D_ASSERT_1, ))
#define OPEN3D_ASSERT(...) \
    OPEN3D_ASSERT_EXPAND(OPEN3D_ASSERT_CHOOSER(__VA_ARGS__)(__VA_ARGS__))
#define OPEN3D_ASSERT_1(condition) \
    OPEN3D_ASSERT_MSG(condition, "Assertion failed: " #condition)

#if defined(__CUDA_ARCH__)

#include "open3d/core/CUDAUtils.h"

#define OPEN3D_ASSERT_MSG(condition, message)                         \
    do {                                                              \
        if (!(condition)) {                                           \
            ::open3d::detail::Open3DCudaAssertReportAndTrap(message); \
        }                                                             \
    } while (0)

#elif defined(__SYCL_DEVICE_ONLY__)

#include <sycl/ext/oneapi/experimental/builtins.hpp>
#include <sycl/sycl.hpp>

namespace open3d {
namespace detail {

inline void Open3DSyclAssertReportAndTrap(const char *message) {
    sycl::ext::oneapi::experimental::printf(
            "Open3D SYCL device assertion failed: %s\n", message);
    __builtin_trap();
}

}  // namespace detail
}  // namespace open3d

#define OPEN3D_ASSERT_MSG(condition, message)                         \
    do {                                                              \
        if (!(condition)) {                                           \
            ::open3d::detail::Open3DSyclAssertReportAndTrap(message); \
        }                                                             \
    } while (0)

#else

#define OPEN3D_ASSERT_MSG(condition, message)                         \
    do {                                                              \
        if (!(condition)) {                                           \
            ::open3d::utility::Logger::LogError_(                     \
                    __FILE__, __LINE__,                               \
                    static_cast<const char *>(OPEN3D_FUNCTION), "{}", \
                    message);                                         \
        }                                                             \
    } while (0)

#endif
