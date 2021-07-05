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
#include "open3d/geometry/KDTreeFlann.h"
#include "open3d/io/PointCloudIO.h"
#include "open3d/t/io/PointCloudIO.h"

namespace open3d {
namespace t {
namespace geometry {

static const std::string path = std::string(TEST_DATA_DIR) + "/fragment.ply";

void HybridSearch(benchmark::State& state,
                  const core::Device& device,
                  double voxel_size,
                  double radius,
                  double max_nn) {
    t::geometry::PointCloud pcd;

    t::io::ReadPointCloud(path, pcd, {"auto", false, false, false});
    pcd = pcd.To(device);

    pcd.VoxelDownSample(voxel_size);

    core::nns::NearestNeighborSearch tree(pcd.GetPoints());

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

void KnnSearch(benchmark::State& state,
               const core::Device& device,
               double voxel_size,
               double max_nn) {
    t::geometry::PointCloud pcd;

    t::io::ReadPointCloud(path, pcd, {"auto", false, false, false});
    pcd = pcd.To(device);

    pcd.VoxelDownSample(voxel_size);

    core::nns::NearestNeighborSearch tree(pcd.GetPoints());

    bool check = tree.KnnIndex();
    if (!check) {
        utility::LogError("NearestNeighborSearch::KNN Index is not set.");
    }

    core::Tensor indices, distance, counts;
    // Warp up
    std::tie(indices, distance) = tree.KnnSearch(pcd.GetPoints(), max_nn);
    for (auto _ : state) {
        std::tie(indices, distance) = tree.KnnSearch(pcd.GetPoints(), max_nn);
    }
}

void LegacySearchForPointCloud(
        const open3d::geometry::PointCloud& pcd,
        const open3d::geometry::KDTreeSearchParam& search_param) {
    open3d::geometry::KDTreeFlann kdtree;

    kdtree.SetGeometry(pcd);
    auto points = pcd.points_;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)points.size(); i++) {
        std::vector<int> indices;
        std::vector<double> distance2;
        kdtree.Search(points[i], search_param, indices, distance2);
    }
}

void LegacySearch(benchmark::State& state,
                  double voxel_size,
                  const open3d::geometry::KDTreeSearchParam& search_param) {
    open3d::geometry::PointCloud pcd;

    open3d::io::ReadPointCloud(path, pcd, {"auto", false, false, false});

    auto pcd_down = pcd.VoxelDownSample(voxel_size);

    LegacySearchForPointCloud(*pcd_down, search_param);
    for (auto _ : state) {
        LegacySearchForPointCloud(*pcd_down, search_param);
    }
}

BENCHMARK_CAPTURE(
        HybridSearch, [0.07 | 30] CPU, core::Device("CPU:0"), 0.02, 0.07, 30)
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(
        HybridSearch, [0.07 | 30] CUDA, core::Device("CUDA:0"), 0.02, 0.07, 30)
        ->Unit(benchmark::kMillisecond);
#endif

BENCHMARK_CAPTURE(
        HybridSearch, [0.04 | 30] CPU, core::Device("CPU:0"), 0.04, 0.07, 30)
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(
        HybridSearch, [0.04 | 30] CUDA, core::Device("CUDA:0"), 0.04, 0.07, 30)
        ->Unit(benchmark::kMillisecond);
#endif

BENCHMARK_CAPTURE(
        HybridSearch, [0.07 | 1] CPU, core::Device("CPU:0"), 0.02, 0.07, 1)
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(
        HybridSearch, [0.07 | 1] CUDA, core::Device("CUDA:0"), 0.02, 0.07, 1)
        ->Unit(benchmark::kMillisecond);
#endif

BENCHMARK_CAPTURE(KnnSearch, [30] CPU, core::Device("CPU:0"), 0.02, 1)
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(KnnSearch, [30] CUDA, core::Device("CUDA:0"), 0.02, 30)
        ->Unit(benchmark::kMillisecond);
#endif

BENCHMARK_CAPTURE(KnnSearch, [1] CPU, core::Device("CPU:0"), 0.02, 1)
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(KnnSearch, [1] CUDA, core::Device("CUDA:0"), 0.02, 1)
        ->Unit(benchmark::kMillisecond);
#endif

BENCHMARK_CAPTURE(
        LegacySearch, [30] KNN, 0.02, open3d::geometry::KDTreeSearchParamKNN())
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(
        LegacySearch, [1] KNN, 0.02, open3d::geometry::KDTreeSearchParamKNN(1))
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(LegacySearch,
                  [0.07 | 30] Hybrid,
                  0.02,
                  open3d::geometry::KDTreeSearchParamHybrid(0.07, 30))
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(LegacySearch,
                  [0.07 | 1] Hybrid,
                  0.02,
                  open3d::geometry::KDTreeSearchParamHybrid(0.07, 1))
        ->Unit(benchmark::kMillisecond);

}  // namespace geometry
}  // namespace t
}  // namespace open3d
