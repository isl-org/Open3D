// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <tbb/parallel_for.h>

#include <Eigen/Dense>

#include "open3d/geometry/KDTreeFlann.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/Parallel.h"
#include "open3d/utility/ProgressBar.h"

namespace open3d {
namespace geometry {

std::vector<int> PointCloud::ClusterDBSCAN(double eps,
                                           size_t min_points,
                                           bool print_progress,
                                           bool precompute_neighbors) const {
    KDTreeFlann kdtree(*this);
    utility::ProgressBar progress_bar(points_.size(), "Clustering",
                                      print_progress);
    std::vector<std::vector<int>> nbs;
    if (precompute_neighbors) {
        utility::LogDebug("Precompute neighbors.");
        progress_bar.Reset(points_.size(), "Precompute neighbors.",
                           print_progress);
        nbs.resize(points_.size());
        tbb::parallel_for(
                tbb::blocked_range<std::size_t>(0, points_.size(),
                                                utility::DefaultGrainSizeTBB()),
                [&](const tbb::blocked_range<std::size_t>& range) {
                    for (std::size_t i = range.begin(); i < range.end(); ++i) {
                        std::vector<double> dists2;
                        kdtree.SearchRadius(points_[i], eps, nbs[i], dists2);
                    }
                    progress_bar += (range.end() - range.begin());
                });
        utility::LogDebug("Done Precompute neighbors.");
    }

    // Reuse one neighborhood in low-memory mode instead of retaining all edges.
    std::vector<int> neighbors;
    std::vector<double> dists2;
    const auto get_neighbors = [&](size_t idx) -> const std::vector<int>& {
        if (precompute_neighbors) {
            return nbs[idx];
        }
        kdtree.SearchRadius(points_[idx], eps, neighbors, dists2);
        return neighbors;
    };

    // Set all labels to undefined (-2).
    utility::LogDebug("Compute Clusters");
    progress_bar.Reset(points_.size(), "Clustering", print_progress);
    std::vector<int> labels(points_.size(), -2);
    std::vector<int> nbs_next;
    int cluster_label = 0;
    for (size_t idx = 0; idx < points_.size(); ++idx) {
        // Label is not undefined.
        if (labels[idx] != -2) {
            continue;
        }

        const auto& seed_neighbors = get_neighbors(idx);
        ++progress_bar;
        if (seed_neighbors.size() < min_points) {
            labels[idx] = -1;
            continue;
        }

        labels[idx] = cluster_label;

        // Label on discovery so each point enters the work-list at most once.
        // Previously visited noise can become a border point, but is not core.
        const auto add_neighbors = [&](const std::vector<int>& indices) {
            for (int nb : indices) {
                if (labels[nb] == -2) {
                    nbs_next.push_back(nb);
                }
                if (labels[nb] < 0) {
                    labels[nb] = cluster_label;
                }
            }
        };
        add_neighbors(seed_neighbors);
        while (!nbs_next.empty()) {
            int nb = nbs_next.back();
            nbs_next.pop_back();

            const auto& current_neighbors = get_neighbors(nb);
            ++progress_bar;
            if (current_neighbors.size() >= min_points) {
                add_neighbors(current_neighbors);
            }
        }

        cluster_label++;
    }

    utility::LogDebug("Done Compute Clusters: {:d}", cluster_label);
    return labels;
}

}  // namespace geometry
}  // namespace open3d
