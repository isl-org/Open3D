// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/utility/ISAInfo.isph"

#if ISPC_POINTER_SIZE != 64
#error Detected platform which is not 64-bit.
#endif

#if ISPC_FP64_SUPPORTED != 1
#error 64-bit float support is not available.
#endif

OPEN3D_DEFINE_ISA_INFO_GETTERS(open3d_utility_)
