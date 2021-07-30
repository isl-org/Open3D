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

#include "open3d/core/nns/NearestNeighborSearch.h"

#include <benchmark/benchmark.h>

#include <string>
#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR
#endif

#include "open3d/core/CUDAUtils.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/io/PointCloudIO.h"
#include "open3d/utility/Console.h"

namespace open3d {
namespace core {

std::pair<Tensor, Tensor> PrepareInput(const Device& device,
                                       const Dtype& dtype,
                                       double voxel_size) {
    // Test data.
    static const std::string source_filename =
            fmt::format("{}/fragment.pcd", std::string(TEST_DATA_DIR));
    static const std::string target_filename =
            fmt::format("{}/fragment.pcd", std::string(TEST_DATA_DIR));
    // Load test point cloud data.
    t::geometry::PointCloud dataset_pc;
    t::geometry::PointCloud query_pc;
    t::io::ReadPointCloud(source_filename, dataset_pc,
                          {"auto", false, false, true});
    t::io::ReadPointCloud(target_filename, query_pc,
                          {"auto", false, false, true});

    if (voxel_size > 0) {
        dataset_pc = dataset_pc.VoxelDownSample(voxel_size);
        query_pc = query_pc.VoxelDownSample(voxel_size);
    }

    // Build Tensor.
    Tensor dataset_points =
            dataset_pc.GetPoints().To(device, dtype, /*copy*/ true);
    Tensor query_points = query_pc.GetPoints().To(device, dtype, /*copy*/ true);

    return std::make_pair(dataset_points, query_points);
}

static void BM_TestNNS_Hybrid(benchmark::State& state, const Device& device) {
    // state.range(n) are arguments that are passed to us
    double radius = state.range(0) / 1000.0;
    int max_knn = state.range(1);

    // Prepare input data.
    Tensor dataset_points, query_points;
    std::tie(dataset_points, query_points) =
            PrepareInput(device, core::Float32, -1);

    // Setup NNS.
    nns::NearestNeighborSearch nns(dataset_points);
    nns.HybridIndex(radius);

    // Search.
    Tensor indices, distances, num_neighbors;
    for (auto _ : state) {
        std::tie(indices, distances, num_neighbors) =
                nns.HybridSearch(query_points, radius, max_knn);
        cuda::Synchronize(device);
    }
}

static void BM_TestNNS_Radius(benchmark::State& state,
                              const Device& device,
                              bool sort) {
    // state.range(n) are arguments that are passed to us
    double radius = state.range(0) / 1000.0;

    // Prepare input data.
    Tensor dataset_points, query_points;
    std::tie(dataset_points, query_points) =
            PrepareInput(device, core::Float32, -1);

    // Setup NNS.
    nns::NearestNeighborSearch nns(dataset_points);
    nns.FixedRadiusIndex(radius);

    // Search.
    Tensor indices, distances, neighbors_row_splits;
    for (auto _ : state) {
        std::tie(indices, distances, neighbors_row_splits) =
                nns.FixedRadiusSearch(query_points, radius, sort);
        cuda::Synchronize(device);
    }
}

static void BM_TestNNS_Knn(benchmark::State& state,
                           const Device& device,
                           double voxel_size) {
    // state.range(n) are arguments that are passed to us
    int knn = state.range(0);

    // Prepare input data.
    Tensor dataset_points, query_points;
    std::tie(dataset_points, query_points) =
            PrepareInput(device, core::Float32, voxel_size);

    // Setup NNS.
    nns::NearestNeighborSearch nns(dataset_points);
    nns.KnnIndex();

    // Search.
    Tensor indices, distances;
    for (auto _ : state) {
        std::tie(indices, distances) = nns.KnnSearch(query_points, knn);
        cuda::Synchronize();
    }
}

BENCHMARK_CAPTURE(BM_TestNNS_Hybrid, CPU, Device("CPU:0"))
        ->Args({100, 1})
        ->Args({100, 64})
        ->Args({100, 256})
        ->Args({200, 1})
        ->Args({200, 64})
        ->Args({200, 256})
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_TestNNS_Radius, CPU, Device("CPU:0"), true)
        ->Args({10})
        ->Args({50})
        ->Args({100})
        ->Args({200})
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_TestNNS_Knn, CPU_without_downsample, Device("CPU:0"), -1.0)
        ->Args({1})
        ->Args({16})
        ->Args({64})
        ->Args({128})
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_TestNNS_Knn, CPU_downsample_0 .005, Device("CPU:0"), 0.005)
        ->Args({1})
        ->Args({16})
        ->Args({64})
        ->Args({128})
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_TestNNS_Knn, CPU_downsample_0 .05, Device("CPU:0"), 0.05)
        ->Args({1})
        ->Args({16})
        ->Args({64})
        ->Args({128})
        ->Unit(benchmark::kMillisecond);
#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(BM_TestNNS_Hybrid, GPU, Device("CUDA:0"))
        ->Args({100, 1})
        ->Args({100, 64})
        ->Args({100, 256})
        ->Args({200, 1})
        ->Args({200, 64})
        ->Args({200, 256})
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_TestNNS_Radius, GPU_SORT, Device("CUDA:0"), true)
        ->Args({10})
        ->Args({50})
        ->Args({100})
        ->Args({200})
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_TestNNS_Radius, GPU_UNSORT, Device("CUDA:0"), false)
        ->Args({10})
        ->Args({50})
        ->Args({100})
        ->Args({200})
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_TestNNS_Knn,
                  GPU_without_downsample,
                  Device("CUDA:0"),
                  -1.0)
        ->Args({1})
        ->Args({16})
        ->Args({64})
        ->Args({128})
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_TestNNS_Knn,
                  GPU_downsample_0 .005,
                  Device("CUDA:0"),
                  0.005)
        ->Args({1})
        ->Args({16})
        ->Args({64})
        ->Args({128})
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_TestNNS_Knn, GPU_downsample_0 .05, Device("CUDA:0"), 0.05)
        ->Args({1})
        ->Args({16})
        ->Args({64})
        ->Args({128})
        ->Unit(benchmark::kMillisecond);
#endif
}  // namespace core
}  // namespace open3d
