// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// This file contains headers for BLAS/LAPACK implementations for CUDA.
//
// For developers, please make sure that this file is not ultimately included in
// Open3D.h.

#pragma once

#ifdef BUILD_CUDA_MODULE
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusolver_common.h>
#endif
