// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
#pragma once

#include <vector>

#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/Geometry.h"
namespace open3d {
namespace t {
namespace geometry {

/// Common code for computing geometry metrics from pairwise point distances.
/// This function expects Euclidean distances as input and returns the requested
/// metrics between point clouds / meshes.
core::Tensor ComputeMetricsCommon(core::Tensor distance12,
                                  core::Tensor distance21,
                                  std::vector<Metric> metrics,
                                  MetricParameters params);
}  // namespace geometry
}  // namespace t
}  // namespace open3d
