// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "open3d/geometry/TriangleMesh.h"

#include <benchmark/benchmark.h>

#include "open3d/data/Dataset.h"
#include "open3d/io/PointCloudIO.h"

namespace open3d {
namespace pipelines {
namespace registration {

// This pcd does not contains non-finite points.
// TODO: Change this to pcd with non-finite poins.
static void BenchmarkCreateFromPointCloudBallPivoting(
        benchmark::State& state, const bool remove_non_finite_points) {
    data::SamplePointCloudPCD sample_pcd;
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
