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

#include "open3d/t/pipelines/registration/Registration.h"

#include <benchmark/benchmark.h>

#include "open3d/core/nns/NearestNeighborSearch.h"
#include "open3d/t/io/PointCloudIO.h"
#include "open3d/t/pipelines/registration/TransformationEstimation.h"

// Testing parameters:
// Filename for pointcloud registration data.
static const std::string source_pointcloud_filename =
        std::string(TEST_DATA_DIR) + "/ICP/cloud_bin_0.pcd";
static const std::string target_pointcloud_filename =
        std::string(TEST_DATA_DIR) + "/ICP/cloud_bin_1.pcd";

static const double voxel_downsampling_factor = 0.01;

// ICP ConvergenceCriteria.
static const double relative_fitness = 1e-6;
static const double relative_rmse = 1e-6;
static const int max_iterations = 10;

// NNS parameter.
static const double max_correspondence_distance = 0.03;

// Initial transformation guess for registation.
static const std::vector<double> initial_transform_flat{
        0.862, 0.011, -0.507, 0.5,  -0.139, 0.967, -0.215, 0.7,
        0.487, 0.255, 0.835,  -1.4, 0.0,    0.0,   0.0,    1.0};

namespace open3d {
namespace t {
namespace pipelines {
namespace registration {

static void BenchmarkRegistrationICP(benchmark::State& state,
                                     const core::Device& device,
                                     const TransformationEstimationType& type) {
    geometry::PointCloud source(device), target(device);

    io::ReadPointCloud(source_pointcloud_filename, source,
                       {"auto", false, false, true});
    io::ReadPointCloud(target_pointcloud_filename, target,
                       {"auto", false, false, true});

    source = source.To(device).VoxelDownSample(voxel_downsampling_factor);
    target = target.To(device).VoxelDownSample(voxel_downsampling_factor);

    std::shared_ptr<TransformationEstimation> estimation;
    if (type == TransformationEstimationType::PointToPlane) {
        estimation = std::make_shared<TransformationEstimationPointToPlane>();
    } else if (type == TransformationEstimationType::PointToPoint) {
        estimation = std::make_shared<TransformationEstimationPointToPoint>();
    }

    core::Tensor init_trans = core::Tensor(initial_transform_flat, {4, 4},
                                           core::Dtype::Float64, device);
    core::Dtype dtype = source.GetPoints().GetDtype();
    init_trans = init_trans.To(dtype);

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
    }

    utility::LogDebug(" PointCloud Size: Source: {}  Target: {}",
                      source.GetPoints().GetShape().ToString(),
                      target.GetPoints().GetShape().ToString());
    utility::LogDebug(" Max iterations: {}, Max_correspondence_distance : {}",
                      max_iterations, max_correspondence_distance);
    utility::LogDebug(" Correspondences: {}, Fitness: {}, Inlier RMSE: {}",
                      reg_result.correspondence_set_.first.GetLength(),
                      reg_result.fitness_, reg_result.inlier_rmse_);
}

BENCHMARK_CAPTURE(BenchmarkRegistrationICP,
                  PointToPlane / CPU,
                  core::Device("CPU:0"),
                  TransformationEstimationType::PointToPlane)
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(BenchmarkRegistrationICP,
                  PointToPoint / CPU,
                  core::Device("CPU:0"),
                  TransformationEstimationType::PointToPoint)
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(BenchmarkRegistrationICP,
                  PointToPlane / CUDA,
                  core::Device("CUDA:0"),
                  TransformationEstimationType::PointToPlane)
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(BenchmarkRegistrationICP,
                  PointToPoint / CUDA,
                  core::Device("CUDA:0"),
                  TransformationEstimationType::PointToPoint)
        ->Unit(benchmark::kMillisecond);
#endif

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
