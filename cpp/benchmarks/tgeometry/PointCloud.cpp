// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#include "open3d/t/geometry/PointCloud.h"

#include <benchmark/benchmark.h>

namespace open3d {
namespace t {
namespace geometry {

void FromLegacyPointCloud(benchmark::State& state, const core::Device& device) {
    open3d::geometry::PointCloud legacy_pcd;
    size_t num_points = 1000000;  // 1M
    legacy_pcd.points_ =
            std::vector<Eigen::Vector3d>(num_points, Eigen::Vector3d(0, 0, 0));
    legacy_pcd.points_ =
            std::vector<Eigen::Vector3d>(num_points, Eigen::Vector3d(0, 0, 0));

    // Warm up.
    t::geometry::PointCloud pcd = t::geometry::PointCloud::FromLegacyPointCloud(
            legacy_pcd, core::Dtype::Float32, device);
    (void)pcd;

    for (auto _ : state) {
        t::geometry::PointCloud pcd =
                t::geometry::PointCloud::FromLegacyPointCloud(
                        legacy_pcd, core::Dtype::Float32, device);
    }
}

BENCHMARK_CAPTURE(FromLegacyPointCloud, CPU, core::Device("CPU:0"))
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(FromLegacyPointCloud, CUDA, core::Device("CUDA:0"))
        ->Unit(benchmark::kMillisecond);
#endif

}  // namespace geometry
}  // namespace t
}  // namespace open3d
