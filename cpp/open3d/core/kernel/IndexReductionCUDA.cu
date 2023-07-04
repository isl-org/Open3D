// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Dispatch.h"
#include "open3d/core/Indexer.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/Tensor.h"

namespace open3d {
namespace core {
namespace kernel {

void IndexMeanCUDA_(const Tensor& index, const Tensor& src, Tensor& dst) {}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
