// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// cppcheck-suppress missingIncludeSystem
#include <Eigen/Core>

// cppcheck-suppress missingIncludeSystem
#include <vector>

#include "open3d/utility/Parallel.h"

namespace open3d {
namespace geometry {
namespace smoothing {

// Applies one Laplacian-style pass that preserves 1:1 point indexing.
template <typename ForEachNeighborFunc, typename ComputeWeightFunc>
void ApplyIndexedLaplacianUpdate(
        const std::vector<Eigen::Vector3d> &reference_positions,
        const std::vector<Eigen::Vector3d> &previous_values,
        std::vector<Eigen::Vector3d> &next_values,
        double factor,
        const ForEachNeighborFunc &for_each_neighbor,
        const ComputeWeightFunc &compute_weight) {
    const int n_values = static_cast<int>(previous_values.size());
    next_values.resize(previous_values.size());

#pragma omp parallel for schedule(static) \
        num_threads(utility::EstimateMaxThreads())
    for (int index = 0; index < n_values; ++index) {
        Eigen::Vector3d weighted_sum = Eigen::Vector3d::Zero();
        double total_weight = 0.0;
        for_each_neighbor(index, [&](int neighbor_index) {
            const double weight =
                    compute_weight(index, neighbor_index, reference_positions);
            total_weight += weight;
            weighted_sum += weight * previous_values[neighbor_index];
        });

        if (total_weight > 0.0) {
            next_values[index] = previous_values[index] +
                                 factor * (weighted_sum / total_weight -
                                           previous_values[index]);
        } else {
            next_values[index] = previous_values[index];
        }
    }
}

}  // namespace smoothing
}  // namespace geometry
}  // namespace open3d
