// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <vector>

#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/Geometry.h"
namespace open3d {
namespace t {
namespace geometry {

std::vector<float> ComputeDistanceCommon(core::Tensor distance12,
                                         core::Tensor distance21,
                                         std::vector<Metric> metrics,
                                         MetricParameters params);
}
}  // namespace t
}  // namespace open3d
