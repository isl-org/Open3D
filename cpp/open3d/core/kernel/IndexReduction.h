// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/core/Tensor.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace kernel {

void IndexSum_(const Tensor& index, const Tensor& src, Tensor& dst);

void IndexSumCPU_(const Tensor& index, const Tensor& src, Tensor& dst);

#ifdef BUILD_CUDA_MODULE
void IndexSumCUDA_(const Tensor& index, const Tensor& src, Tensor& dst);
#endif

}  // namespace kernel
}  // namespace core
}  // namespace open3d
