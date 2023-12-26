// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/core/Tensor.h"
#include "open3d/utility/Optional.h"

namespace open3d {
namespace t {

namespace geometry {
class PointCloud;
}

namespace pipelines {
namespace registration {

/// Function to compute FPFH feature for a point cloud.
/// It uses KNN search (Not recommended to use on GPU) if only max_nn parameter
/// is provided, Radius search (Not recommended to use on GPU) if only radius
/// parameter is provided, and Hybrid search (Recommended) if both are provided.
///
/// \param input The input point cloud with data type float32 or float64.
/// \param max_nn [optional] Neighbor search max neighbors parameter. [Default =
/// 100].
/// \param radius [optional] Neighbor search radius parameter. [Recommended ~5x
/// voxel size].
/// \return A Tensor of FPFH feature of the input point cloud with
/// shape {N, 33}, data type and device same as input.
core::Tensor ComputeFPFHFeature(
        const geometry::PointCloud &input,
        const utility::optional<int> max_nn = 100,
        const utility::optional<double> radius = utility::nullopt);

/// \brief Function to find correspondences via 1-nearest neighbor feature
/// matching. Target is used to construct a nearest neighbor search
/// object, in order to query source.
/// \param source_feats (N, D) tensor
/// \param target_feats (M, D) tensor
/// \param mutual_filter Boolean flag, only return correspondences (i, j) s.t.
/// source_features[i] and target_features[j] are mutually the nearest neighbor.
/// \param mutual_consistency_ratio Float threshold to decide whether the number
/// of correspondences is sufficient. Only used when mutual_filter is set to
/// True.
/// \return (K, 2, Int64) tensor. When mutual_filter is disabled: the first
/// column is arange(0, N) of source, and the second column is the corresponding
/// index of target. When mutual_filter is enabled, return the filtering subset
/// of the aforementioned correspondence set where source[i] and target[j] are
/// mutually the nearest neighbor. If the subset size is smaller than
/// mutual_consistency_ratio * N, return the unfiltered set.
core::Tensor CorrespondencesFromFeatures(const core::Tensor &source_features,
                                         const core::Tensor &target_features,
                                         bool mutual_filter = false,
                                         float mutual_consistency_ratio = 0.1);
}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
