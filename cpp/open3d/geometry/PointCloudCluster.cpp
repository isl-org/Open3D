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
                                           bool print_progress) const {
    KDTreeFlann kdtree(*this);

    // Precompute all neighbors.
    utility::LogDebug("Precompute neighbors.");
    utility::ProgressBar progress_bar(points_.size(), "Precompute neighbors.",
                                      print_progress);
    std::vector<std::vector<int>> nbs(points_.size());

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

    // Set all labels to undefined (-2).
    utility::LogDebug("Compute Clusters");
    progress_bar.Reset(points_.size(), "Clustering", print_progress);
    std::vector<int> labels(points_.size(), -2);
    int cluster_label = 0;
    for (size_t idx = 0; idx < points_.size(); ++idx) {
        // Label is not undefined.
        if (labels[idx] != -2) {
            continue;
        }

        // Check density.
        if (nbs[idx].size() < min_points) {
            labels[idx] = -1;
            continue;
        }

        labels[idx] = cluster_label;
        ++progress_bar;

        // BFS/DFS expansion from the seed point: push all neighbors of idx as
        // the initial work-list, then iteratively pop, label unlabeled points,
        // and push their neighbors if they are core points. Duplicates are
        // harmlessly skipped by the labels[nb] >= 0 check.
        std::vector<int> nbs_next(nbs[idx].begin(), nbs[idx].end());

        while (!nbs_next.empty()) {
            int nb = nbs_next.back();
            nbs_next.pop_back();

            // Skip the seed point and already-labeled neighbors.
            if (nb == int(idx) || labels[nb] >= 0) {
                continue;
            }
            labels[nb] = cluster_label;
            ++progress_bar;

            // Expand from core points.
            if (nbs[nb].size() >= min_points) {
                for (int qnb : nbs[nb]) {
                    if (labels[qnb] < 0) {
                        nbs_next.push_back(qnb);
                    }
                }
            }
        }

        cluster_label++;
    }

    utility::LogDebug("Done Compute Clusters: {:d}", cluster_label);
    return labels;
}

}  // namespace geometry
}  // namespace open3d
