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

#include "open3d/t/pipelines/registration/Registration.h"

#include <benchmark/benchmark.h>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/nns/NearestNeighborSearch.h"
#include "open3d/t/io/PointCloudIO.h"
#include "open3d/t/pipelines/registration/TransformationEstimation.h"

static const double voxel_downsampling_factor = 0.02;

// ICP ConvergenceCriteria.
static const double relative_fitness = 1e-6;
static const double relative_rmse = 1e-6;
static const int max_iterations = 10;

// NNS parameter.
static const double max_correspondence_distance = 0.05;

// Initial transformation guess for registation.
static const std::vector<float> initial_transform_flat{
        0.862, 0.011, -0.507, 0.5,  -0.139, 0.967, -0.215, 0.7,
        0.487, 0.255, 0.835,  -1.4, 0.0,    0.0,   0.0,    1.0};

namespace open3d {
namespace t {
namespace pipelines {
namespace registration {

static std::tuple<geometry::PointCloud, geometry::PointCloud>
LoadTensorPointCloudFromFile(const std::string& source_pointcloud_filename,
                             const std::string& target_pointcloud_filename,
                             const double voxel_downsample_factor,
                             const core::Dtype& dtype,
                             const core::Device& device) {
    geometry::PointCloud source, target;

    io::ReadPointCloud(source_pointcloud_filename, source,
                       {"auto", false, false, true});
    io::ReadPointCloud(target_pointcloud_filename, target,
                       {"auto", false, false, true});

    source = source.To(device);
    target = target.To(device);

    // Eliminates the case of impractical values (including negative).
    if (voxel_downsample_factor > 0.001) {
        source = source.VoxelDownSample(voxel_downsample_factor);
        target = target.VoxelDownSample(voxel_downsample_factor);
    } else {
        utility::LogWarning(
                "VoxelDownsample: Impractical voxel size [< 0.001], skiping "
                "downsampling.");
    }

    source.SetPointPositions(source.GetPointPositions().To(dtype));
    source.SetPointNormals(source.GetPointNormals().To(dtype));
    if (source.HasPointColors()) {
        source.SetPointColors(source.GetPointColors().To(dtype).Div(255.0));
    }

    target.SetPointPositions(target.GetPointPositions().To(dtype));
    target.SetPointNormals(target.GetPointNormals().To(dtype));
    if (target.HasPointColors()) {
        target.SetPointColors(target.GetPointColors().To(dtype).Div(255.0));
    }

    return std::make_tuple(source, target);
}

static void BenchmarkRegistrationICP(benchmark::State& state,
                                     const core::Device& device,
                                     const core::Dtype& dtype,
                                     const TransformationEstimationType& type) {
    geometry::PointCloud source(device), target(device);

    std::string source_pointcloud_filename;
    std::string target_pointcloud_filename;

    core::Tensor init_trans;
    std::shared_ptr<TransformationEstimation> estimation;
    if (type == TransformationEstimationType::PointToPlane) {
        estimation = std::make_shared<TransformationEstimationPointToPlane>();

        init_trans = core::Tensor(initial_transform_flat, {4, 4}, core::Float32,
                                  device)
                             .To(dtype);

        source_pointcloud_filename =
                std::string(TEST_DATA_DIR) + "/ICP/cloud_bin_0.pcd";
        target_pointcloud_filename =
                std::string(TEST_DATA_DIR) + "/ICP/cloud_bin_1.pcd";
    } else if (type == TransformationEstimationType::PointToPoint) {
        estimation = std::make_shared<TransformationEstimationPointToPoint>();

        init_trans = core::Tensor(initial_transform_flat, {4, 4}, core::Float32,
                                  device)
                             .To(dtype);

        source_pointcloud_filename =
                std::string(TEST_DATA_DIR) + "/ICP/cloud_bin_0.pcd";
        target_pointcloud_filename =
                std::string(TEST_DATA_DIR) + "/ICP/cloud_bin_1.pcd";
    } else if (type == TransformationEstimationType::ColoredICP) {
        estimation = std::make_shared<TransformationEstimationForColoredICP>();

        init_trans = core::Tensor::Eye(4, core::Float64, device);

        source_pointcloud_filename =
                std::string(TEST_DATA_DIR) + "/ColoredICP/frag_115.ply";
        target_pointcloud_filename =
                std::string(TEST_DATA_DIR) + "/ColoredICP/frag_115.ply";
    }

    std::tie(source, target) = LoadTensorPointCloudFromFile(
            source_pointcloud_filename, target_pointcloud_filename,
            voxel_downsampling_factor, dtype, device);

    RegistrationResult reg_result(init_trans);

    // Warm up.
    reg_result = RegistrationICP(
            source, target, max_correspondence_distance, init_trans,
            *estimation,
            ICPConvergenceCriteria(relative_fitness, relative_rmse,
                                   max_iterations));

    for (auto _ : state) {
        reg_result = RegistrationICP(
                source, target, max_correspondence_distance, init_trans,
                *estimation,
                ICPConvergenceCriteria(relative_fitness, relative_rmse,
                                       max_iterations));
        core::cuda::Synchronize(device);
    }
}

BENCHMARK_CAPTURE(BenchmarkRegistrationICP,
                  PointToPlane / CPU32,
                  core::Device("CPU:0"),
                  core::Float32,
                  TransformationEstimationType::PointToPlane)
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(BenchmarkRegistrationICP,
                  PointToPlane / CUDA32,
                  core::Device("CUDA:0"),
                  core::Float32,
                  TransformationEstimationType::PointToPlane)
        ->Unit(benchmark::kMillisecond);
#endif

BENCHMARK_CAPTURE(BenchmarkRegistrationICP,
                  PointToPlane / CPU64,
                  core::Device("CPU:0"),
                  core::Float64,
                  TransformationEstimationType::PointToPlane)
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(BenchmarkRegistrationICP,
                  PointToPlane / CUDA64,
                  core::Device("CUDA:0"),
                  core::Float64,
                  TransformationEstimationType::PointToPlane)
        ->Unit(benchmark::kMillisecond);
#endif

BENCHMARK_CAPTURE(BenchmarkRegistrationICP,
                  PointToPoint / CPU32,
                  core::Device("CPU:0"),
                  core::Float32,
                  TransformationEstimationType::PointToPoint)
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(BenchmarkRegistrationICP,
                  PointToPoint / CUDA32,
                  core::Device("CUDA:0"),
                  core::Float32,
                  TransformationEstimationType::PointToPoint)
        ->Unit(benchmark::kMillisecond);
#endif

BENCHMARK_CAPTURE(BenchmarkRegistrationICP,
                  PointToPoint / CPU64,
                  core::Device("CPU:0"),
                  core::Float64,
                  TransformationEstimationType::PointToPoint)
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(BenchmarkRegistrationICP,
                  PointToPoint / CUDA64,
                  core::Device("CUDA:0"),
                  core::Float64,
                  TransformationEstimationType::PointToPoint)
        ->Unit(benchmark::kMillisecond);
#endif

BENCHMARK_CAPTURE(BenchmarkRegistrationICP,
                  ColoredICP / CPU32,
                  core::Device("CPU:0"),
                  core::Float32,
                  TransformationEstimationType::ColoredICP)
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(BenchmarkRegistrationICP,
                  ColoredICP / CUDA32,
                  core::Device("CUDA:0"),
                  core::Float32,
                  TransformationEstimationType::ColoredICP)
        ->Unit(benchmark::kMillisecond);
#endif

BENCHMARK_CAPTURE(BenchmarkRegistrationICP,
                  ColoredICP / CPU64,
                  core::Device("CPU:0"),
                  core::Float64,
                  TransformationEstimationType::ColoredICP)
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(BenchmarkRegistrationICP,
                  ColoredICP / CUDA64,
                  core::Device("CUDA:0"),
                  core::Float64,
                  TransformationEstimationType::ColoredICP)
        ->Unit(benchmark::kMillisecond);
#endif

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
