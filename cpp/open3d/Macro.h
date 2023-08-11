// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <cassert>

// https://gcc.gnu.org/wiki/Visibility updated to use C++11 attribute syntax
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

// Assertion for CUDA device code.
// Usage:
//     OPEN3D_ASSERT(condition);
//     OPEN3D_ASSERT(condition && "Error message");
// For host-only code, consider using utility::LogError();
#define OPEN3D_ASSERT(...) assert((__VA_ARGS__))
