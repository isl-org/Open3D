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

#include <benchmark/benchmark.h>

#include "open3d/core/Tensor.h"
#include "open3d/core/nns/NearestNeighborSearch.h"
#include "open3d/t/io/PointCloudIO.h"

namespace open3d {
namespace t {
namespace geometry {

static const std::string path = std::string(TEST_DATA_DIR) + "/fragment.ply";

void HybridSearch(benchmark::State& state, const core::Device& device) {
    t::geometry::PointCloud pcd;

    t::io::ReadPointCloud(path, pcd, {"auto", false, false, false});
    pcd = pcd.To(device);

    pcd.VoxelDownSample(0.02);

    core::nns::NearestNeighborSearch tree(pcd.GetPoints());

    double radius = 0.07;
    int max_nn = 30;

    bool check = tree.HybridIndex(radius);
    if (!check) {
        utility::LogError(
                "NearestNeighborSearch::FixedRadiusIndex Index is not set.");
    }

    core::Tensor indices, distance, counts;
    // Warp up
    std::tie(indices, distance, counts) =
            tree.HybridSearch(pcd.GetPoints(), radius, max_nn);
    for (auto _ : state) {
        std::tie(indices, distance, counts) =
                tree.HybridSearch(pcd.GetPoints(), radius, max_nn);
    }
}

BENCHMARK_CAPTURE(HybridSearch, CPU, core::Device("CPU:0"))
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(HybridSearch, CUDA, core::Device("CUDA:0"))
        ->Unit(benchmark::kMillisecond);
#endif

}  // namespace geometry
}  // namespace t
}  // namespace open3d
