#pragma once

#define DISABLE_COPY_AND_ASSIGN(classname) \
    classname(const classname&) = delete;  \
    classname& operator=(const classname&) = delete

#ifdef _WIN32
#if defined(OPEN3D_BUILD_SHARED_LIBS)
#define OPEN3D_EXPORT __declspec(dllexport)
#define OPEN3D_IMPORT __declspec(dllimport)
#else
#define OPEN3D_EXPORT
#define OPEN3D_IMPORT
#endif
#else  // _WIN32
#if defined(__GNUC__)
#define OPEN3D_EXPORT __attribute__((__visibility__("default")))
#else  // defined(__GNUC__)
#define OPEN3D_EXPORT
#endif  // defined(__GNUC__)
#define OPEN3D_IMPORT OPEN3D_EXPORT
#endif  // _WIN32

#define OPEN3D_CONCATENATE_IMPL(s1, s2) s1##s2
#define OPEN3D_CONCATENATE(s1, s2) OPEN3D_CONCATENATE_IMPL(s1, s2)
#define OPEN3D_ANONYMOUS_VARIABLE(str) OPEN3D_CONCATENATE(str, __LINE__)
