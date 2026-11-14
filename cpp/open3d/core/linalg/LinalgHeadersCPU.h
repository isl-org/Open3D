// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// This file contains headers for BLAS/LAPACK implementations. Currently we
// support int64_t interface of OpenBLAS or Intel MKL.
//
// For developers, please make sure that this file is not ultimately included in
// Open3D.h.

#pragma once

#ifdef USE_BLAS
#define OPEN3D_CPU_LINALG_INT int32_t
#define lapack_int int32_t
#include <cblas.h>

// LAPACKE is not available on Windows ARM64 in our vcpkg-based toolchain.
// Build in BLAS-only mode and disable LAPACKE-dependent routines.
#if defined(_WIN32) && defined(_M_ARM64)
#define OPEN3D_DISABLE_LAPACKE 1
#else
#include <lapacke.h>
#endif
#else
#include <mkl.h>
#define OPEN3D_CPU_LINALG_INT MKL_INT
#endif
