// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/core/Tensor.h"

namespace open3d {
namespace core {
namespace kernel {

Tensor NonZero(const Tensor& src);

Tensor NonZeroCPU(const Tensor& src);

#ifdef BUILD_CUDA_MODULE
Tensor NonZeroCUDA(const Tensor& src);
#endif

}  // namespace kernel
}  // namespace core
}  // namespace open3d
