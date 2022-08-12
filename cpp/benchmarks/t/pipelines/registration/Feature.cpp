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

#include "open3d/t/pipelines/registration/Feature.h"

#include <benchmark/benchmark.h>

#include "open3d/core/CUDAUtils.h"
#include "open3d/data/Dataset.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/pipelines/registration/Feature.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/io/PointCloudIO.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace registration {

data::BunnyMesh pointcloud_ply;
static const std::string path = pointcloud_ply.GetPath();

void LegacyComputeFPFHFeature(benchmark::State& state,
                              int max_nn,
                              double radius) {
    auto pcd = open3d::io::CreatePointCloudFromFile(path);
    pcd->EstimateNormals();
    for (auto _ : state) {
        auto fpfh = open3d::pipelines::registration::ComputeFPFHFeature(
                *pcd,
                open3d::geometry::KDTreeSearchParamHybrid(radius, max_nn));
    }
}

void ComputeFPFHFeature(benchmark::State& state,
                        const core::Device& device,
                        const core::Dtype& dtype,
                        int max_nn,
                        double radius) {
    t::geometry::PointCloud pcd;
    t::io::ReadPointCloud(path, pcd);
    pcd = pcd.To(device);
    pcd.SetPointPositions(pcd.GetPointPositions().To(dtype));
    pcd.EstimateNormals();

    core::Tensor fpfh;
    // Warm up.
    fpfh = t::pipelines::registration::ComputeFPFHFeature(pcd, max_nn, radius);

    for (auto _ : state) {
        fpfh = t::pipelines::registration::ComputeFPFHFeature(pcd, max_nn,
                                                              radius);
    }
}

BENCHMARK_CAPTURE(LegacyComputeFPFHFeature,
                  Legacy Hybrid[0.01 | 100],
                  100,
                  0.01)
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(ComputeFPFHFeature,
                  CPU F32 Hybrid[0.01 | 100],
                  core::Device("CPU:0"),
                  core::Float32,
                  100,
                  0.01)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(ComputeFPFHFeature,
                  CPU F64 Hybrid[0.01 | 100],
                  core::Device("CPU:0"),
                  core::Float64,
                  100,
                  0.02)
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(ComputeFPFHFeature,
                  CUDA F32 Hybrid[0.01 | 100],
                  core::Device("CUDA:0"),
                  core::Float32,
                  100,
                  0.02)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(ComputeFPFHFeature,
                  CUDA F64 Hybrid[0.01 | 100],
                  core::Device("CUDA:0"),
                  core::Float64,
                  100,
                  0.02)
        ->Unit(benchmark::kMillisecond);
#endif

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
