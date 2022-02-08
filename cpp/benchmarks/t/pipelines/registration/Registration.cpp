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
#include "open3d/data/Dataset.h"
#include "open3d/t/io/PointCloudIO.h"
#include "open3d/t/pipelines/registration/TransformationEstimation.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace registration {

// Testing parameters:
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

static void BenchmarkICP(benchmark::State& state,
                         const core::Device& device,
                         const core::Dtype& dtype,
                         const TransformationEstimationType& type) {
    utility::SetVerbosityLevel(utility::VerbosityLevel::Error);
    data::DemoICPPointClouds demo_icp_pointclouds;
    geometry::PointCloud source, target;
    std::tie(source, target) = LoadTensorPointCloudFromFile(
            demo_icp_pointclouds.GetPaths(0), demo_icp_pointclouds.GetPaths(1),
            /*voxel_downsampling_factor =*/0.02, dtype, device);

    std::shared_ptr<TransformationEstimation> estimation;
    if (type == TransformationEstimationType::PointToPlane) {
        estimation = std::make_shared<TransformationEstimationPointToPlane>();
    } else if (type == TransformationEstimationType::PointToPoint) {
        estimation = std::make_shared<TransformationEstimationPointToPoint>();
    } else if (type == TransformationEstimationType::ColoredICP) {
        estimation = std::make_shared<TransformationEstimationForColoredICP>();
    }

    core::Tensor init_trans =
            core::Tensor(initial_transform_flat, {4, 4}, core::Float32, device)
                    .To(dtype);

    RegistrationResult reg_result(init_trans);

    // Warm up.
    reg_result = ICP(source, target, max_correspondence_distance, init_trans,
                     *estimation,
                     ICPConvergenceCriteria(relative_fitness, relative_rmse,
                                            max_iterations));

    for (auto _ : state) {
        reg_result = ICP(source, target, max_correspondence_distance,
                         init_trans, *estimation,
                         ICPConvergenceCriteria(relative_fitness, relative_rmse,
                                                max_iterations));
        core::cuda::Synchronize(device);
    }
}

#define ENUM_ICP_METHOD_DEVICE(METHOD_NAME, TRANSFORMATION_TYPE, DEVICE) \
    BENCHMARK_CAPTURE(BenchmarkICP, DEVICE METHOD_NAME##_Float32,        \
                      core::Device(DEVICE), core::Float32,               \
                      TRANSFORMATION_TYPE)                               \
            ->Unit(benchmark::kMillisecond);                             \
    BENCHMARK_CAPTURE(BenchmarkICP, DEVICE METHOD_NAME##_Float64,        \
                      core::Device(DEVICE), core::Float64,               \
                      TRANSFORMATION_TYPE)                               \
            ->Unit(benchmark::kMillisecond);

ENUM_ICP_METHOD_DEVICE(PointToPoint,
                       TransformationEstimationType::PointToPoint,
                       "CPU:0")
ENUM_ICP_METHOD_DEVICE(PointToPlane,
                       TransformationEstimationType::PointToPlane,
                       "CPU:0")
ENUM_ICP_METHOD_DEVICE(ColoredICP,
                       TransformationEstimationType::ColoredICP,
                       "CPU:0")

#ifdef BUILD_CUDA_MODULE
ENUM_ICP_METHOD_DEVICE(PointToPoint,
                       TransformationEstimationType::PointToPoint,
                       "CUDA:0")
ENUM_ICP_METHOD_DEVICE(PointToPlane,
                       TransformationEstimationType::PointToPlane,
                       "CUDA:0")
ENUM_ICP_METHOD_DEVICE(ColoredICP,
                       TransformationEstimationType::ColoredICP,
                       "CUDA:0")
#endif

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
