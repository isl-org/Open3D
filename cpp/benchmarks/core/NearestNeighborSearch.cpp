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

#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/io/PointCloudIO.h"

static const std::string source_filename =
        fmt::format("{}/ICP/cloud_bin_0.pcd", std::string(TEST_DATA_DIR));
static const std::string target_filename =
        fmt::format("{}/ICP/cloud_bin_1.pcd", std::string(TEST_DATA_DIR));

namespace open3d {
namespace benchmarks {

class TestNNS {
    std::unique_ptr<core::nns::NearestNeighborSearch> nns_;

    int pos_ = 0;
    int size_ = 0;

public:
    void setup(core::Tensor& dataset_points, double radius) {
        nns_.reset(new core::nns::NearestNeighborSearch(dataset_points));
        nns_->HybridIndex(radius);
    }

    void search(core::Tensor& query_points, double radius, int max_knn) {
        core::Tensor indices;
        core::Tensor distances;

        std::tie(indices, distances) =
                nns_->HybridSearch(query_points, radius, max_knn);
    }
};
// reuse the same instance so we don't recreate the kdtree every time
TestNNS testNNS;

static void BM_TestNNS(benchmark::State& state, const core::Device& device) {
    // state.range(n) are arguments that are passed to us
    double radius = state.range(0) / 1000.0;
    int max_knn = state.range(1);

    t::geometry::PointCloud dataset_pc;
    t::geometry::PointCloud query_pc;

    t::io::ReadPointCloud(source_filename, dataset_pc,
                          {"auto", false, false, true});
    t::io::ReadPointCloud(target_filename, query_pc,
                          {"auto", false, false, true});

    core::Tensor dataset_points = dataset_pc.GetPoints().To(
            device, core::Dtype::Float32, /*copy*/ true);
    core::Tensor query_points = query_pc.GetPoints().To(
            device, core::Dtype::Float32, /*copy*/ true);

    testNNS.setup(dataset_points, radius);
    for (auto _ : state) {
        testNNS.search(query_points, radius, max_knn);
    }
}
// a few specific sized tests, each ->Args({params}) will be a test
BENCHMARK_CAPTURE(BM_TestNNS, CPU, core::Device("CPU:0"))
        ->Args({1000, 1})
        ->Args({1000, 5})
        ->Unit(benchmark::kMillisecond);
// let benchmark vary parameters for us; run each test only for 0.1sec so it
// doesn't take too long
#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(BM_TestNNS, GPU, core::Device("CUDA:0"))
        ->Args({1000, 1})
        ->Args({1000, 5})
        ->Unit(benchmark::kMillisecond);
#endif
}  // namespace benchmarks
}  // namespace open3d
