// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
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
#include <lapacke.h>
#else
#include <mkl.h>
#define OPEN3D_CPU_LINALG_INT MKL_INT
#endif
