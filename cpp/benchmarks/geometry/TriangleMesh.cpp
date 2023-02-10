// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/geometry/TriangleMesh.h"

#include <benchmark/benchmark.h>

#include "open3d/data/Dataset.h"
#include "open3d/io/PointCloudIO.h"

namespace open3d {
namespace pipelines {
namespace registration {

// This pcd does not contains non-finite points.
// TODO: Change this to pcd with non-finite points.
static void BenchmarkCreateFromPointCloudBallPivoting(
        benchmark::State& state, const bool remove_non_finite_points) {
    data::PCDPointCloud sample_pcd;
    auto pcd = io::CreatePointCloudFromFile(sample_pcd.GetPath());

    if (remove_non_finite_points) {
        pcd->RemoveNonFinitePoints();
    }

    std::vector<double> distance = pcd->ComputeNearestNeighborDistance();
    size_t n = distance.size();
    double dist_average = 0.0;
    if (n != 0) {
        dist_average =
                std::accumulate(distance.begin(), distance.end(), 0.0) / n;
    }
    double radius = 1.5 * dist_average;
    std::vector<double> radii = {radius, radius * 1};
    std::shared_ptr<geometry::TriangleMesh> mesh;

    mesh = geometry::TriangleMesh::CreateFromPointCloudBallPivoting(*pcd,
                                                                    radii);

    for (auto _ : state) {
        mesh = geometry::TriangleMesh::CreateFromPointCloudBallPivoting(*pcd,
                                                                        radii);
    }
}

BENCHMARK_CAPTURE(BenchmarkCreateFromPointCloudBallPivoting,
                  Without Non Finite Points,
                  /*remove_non_finite_points*/ true)
        ->Unit(benchmark::kMillisecond);

// TODO: Add BENCHMARK for case `With Non Finite Points`.

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
