// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/core/Dtype.h"
#include "open3d/core/Scalar.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"

namespace open3d {
namespace benchmarks {

/// Returns a Tensor with random values within the range \p range .
core::Tensor Rand(const core::SizeVector& shape,
                  size_t seed,
                  const std::pair<core::Scalar, core::Scalar>& range,
                  core::Dtype dtype,
                  const core::Device& device = core::Device("CPU:0"));

}  // namespace benchmarks
}  // namespace open3d
