// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/core/Tensor.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace pcapartition {

/// Partition the point cloud by recursively doing PCA.
/// \param points Points tensor with shape (N,3).
/// \param max_points The maximum allowed number of points in a partition.
/// \return The number of partitions and an int32 tensor with the partition id
/// for each point. The output tensor uses always the CPU device.
std::tuple<int, core::Tensor> PCAPartition(core::Tensor& points,
                                           int max_points);

}  // namespace pcapartition
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
