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
static_assert(
        sizeof(MKL_INT) == 8,
        "MKL_INT must be 8 bytes: please link with MKL 64-bit int library.");
#define OPEN3D_CPU_LINALG_INT MKL_INT
#endif
