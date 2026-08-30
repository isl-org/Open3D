// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

// cppcheck-suppress missingIncludeSystem
#include <Eigen/Core>

// cppcheck-suppress missingIncludeSystem
#include <vector>

#include "open3d/utility/Parallel.h"

namespace open3d {
namespace geometry {
namespace smoothing {

// Shared explicit Laplacian update used by mesh and point-cloud smoothers.
// For each index i: x_i' = x_i + factor * (sum_j w_ij x_j / sum_j w_ij - x_i).
// Neighbors and weights are supplied by the caller; vertices with no neighbors
// are unchanged.
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

    tbb::parallel_for(
            tbb::blocked_range<int>(0, n_values,
                                    utility::DefaultGrainSizeTBB()),
            [&](const tbb::blocked_range<int> &range) {
                for (int index = range.begin(); index < range.end(); ++index) {
                    Eigen::Vector3d weighted_sum = Eigen::Vector3d::Zero();
                    double total_weight = 0.0;
                    for_each_neighbor(index, [&](int neighbor_index) {
                        const double weight = compute_weight(
                                index, neighbor_index, reference_positions);
                        total_weight += weight;
                        weighted_sum +=
                                weight * previous_values[neighbor_index];
                    });

                    if (total_weight > 0.0) {
                        next_values[index] =
                                previous_values[index] +
                                factor * (weighted_sum / total_weight -
                                          previous_values[index]);
                    } else {
                        next_values[index] = previous_values[index];
                    }
                }
            });
}

}  // namespace smoothing
}  // namespace geometry
}  // namespace open3d
