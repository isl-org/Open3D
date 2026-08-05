// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/pipelines/registration/Registration.h"

#include <benchmark/benchmark.h>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/EigenConverter.h"
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

static const double voxel_downsampling_factor = 0.02;

// NNS parameter.
static const double max_correspondence_distance = 0.05;

// Normal estimation parameters.
static const double normals_search_radius = 10.0;
static const int normals_max_neighbors = 30;

// Initial transformation guess for registration.
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
                "VoxelDownsample: Impractical voxel size [< 0.001], skipping "
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
            voxel_downsampling_factor, dtype, device);

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

core::Tensor ComputeDirectionVectors(const core::Tensor& positions) {
    core::Tensor directions = core::Tensor::Empty(
            positions.GetShape(), positions.GetDtype(), positions.GetDevice());
    for (int64_t i = 0; i < positions.GetLength(); ++i) {
        // Compute the norm of the position vector.
        core::Tensor norm = (positions[i][0] * positions[i][0] +
                             positions[i][1] * positions[i][1] +
                             positions[i][2] * positions[i][2])
                                    .Sqrt();

        // If the norm is zero, set the direction vector to zero.
        if (norm.Item<float>() == 0.0) {
            directions[i].Fill(0.0);
        } else {
            // Otherwise, compute the direction vector by dividing the position
            // vector by its norm.
            directions[i] = positions[i] / norm;
        }
    }
    return directions;
}

static std::tuple<geometry::PointCloud, geometry::PointCloud>
LoadTensorDopplerPointCloudFromFile(
        const std::string& source_pointcloud_filename,
        const std::string& target_pointcloud_filename,
        const double voxel_downsample_factor,
        const core::Dtype& dtype,
        const core::Device& device) {
    geometry::PointCloud source, target;

    io::ReadPointCloud(source_pointcloud_filename, source,
                       {"auto", false, false, true});
    io::ReadPointCloud(target_pointcloud_filename, target,
                       {"auto", false, false, true});

    source.SetPointAttr("directions",
                        ComputeDirectionVectors(source.GetPointPositions()));

    source = source.To(device);
    target = target.To(device);

    // Eliminates the case of impractical values (including negative).
    if (voxel_downsample_factor > 0.001) {
        source = source.VoxelDownSample(voxel_downsample_factor);
        target = target.VoxelDownSample(voxel_downsample_factor);
    } else {
        utility::LogWarning(
                "VoxelDownsample: Impractical voxel size [< 0.001], skipping "
                "downsampling.");
    }

    source.SetPointPositions(source.GetPointPositions().To(dtype));
    source.SetPointAttr("dopplers", source.GetPointAttr("dopplers").To(dtype));
    source.SetPointAttr("directions",
                        source.GetPointAttr("directions").To(dtype));

    target.SetPointPositions(target.GetPointPositions().To(dtype));
    target.EstimateNormals(normals_search_radius, normals_max_neighbors);

    return std::make_tuple(source, target);
}

static void BenchmarkDopplerICP(benchmark::State& state,
                                const core::Device& device,
                                const core::Dtype& dtype,
                                const TransformationEstimationType& type) {
    utility::SetVerbosityLevel(utility::VerbosityLevel::Error);
    data::DemoDopplerICPSequence demo_sequence;
    geometry::PointCloud source, target;
    std::tie(source, target) = LoadTensorDopplerPointCloudFromFile(
            demo_sequence.GetPath(0), demo_sequence.GetPath(1),
            voxel_downsampling_factor, dtype, device);

    Eigen::Matrix4d calibration{Eigen::Matrix4d::Identity()};
    double period{0.0};
    demo_sequence.GetCalibration(calibration, period);

    const core::Tensor calibration_t =
            core::eigen_converter::EigenMatrixToTensor(calibration)
                    .To(device, dtype);

    TransformationEstimationForDopplerICP estimation_dicp;
    estimation_dicp.period_ = period;
    estimation_dicp.transform_vehicle_to_sensor_ = calibration_t;

    core::Tensor init_trans = core::Tensor::Eye(4, dtype, device);
    RegistrationResult reg_result(init_trans);

    // Warm up.
    reg_result = ICP(source, target, max_correspondence_distance, init_trans,
                     estimation_dicp,
                     ICPConvergenceCriteria(relative_fitness, relative_rmse,
                                            max_iterations));

    for (auto _ : state) {
        reg_result = ICP(source, target, max_correspondence_distance,
                         init_trans, estimation_dicp,
                         ICPConvergenceCriteria(relative_fitness, relative_rmse,
                                                max_iterations));
        core::cuda::Synchronize(device);
    }
}

#define ENUM_ICP_METHOD_DEVICE(BENCHMARK_FUNCTION, METHOD_NAME,         \
                               TRANSFORMATION_TYPE, DEVICE)             \
    BENCHMARK_CAPTURE(BENCHMARK_FUNCTION, DEVICE METHOD_NAME##_Float32, \
                      core::Device(DEVICE), core::Float32,              \
                      TRANSFORMATION_TYPE)                              \
            ->Unit(benchmark::kMillisecond);                            \
    BENCHMARK_CAPTURE(BENCHMARK_FUNCTION, DEVICE METHOD_NAME##_Float64, \
                      core::Device(DEVICE), core::Float64,              \
                      TRANSFORMATION_TYPE)                              \
            ->Unit(benchmark::kMillisecond);

ENUM_ICP_METHOD_DEVICE(BenchmarkICP,
                       PointToPoint,
                       TransformationEstimationType::PointToPoint,
                       "CPU:0")
ENUM_ICP_METHOD_DEVICE(BenchmarkICP,
                       PointToPlane,
                       TransformationEstimationType::PointToPlane,
                       "CPU:0")
ENUM_ICP_METHOD_DEVICE(BenchmarkICP,
                       ColoredICP,
                       TransformationEstimationType::ColoredICP,
                       "CPU:0")
ENUM_ICP_METHOD_DEVICE(BenchmarkDopplerICP,
                       DopplerICP,
                       TransformationEstimationType::DopplerICP,
                       "CPU:0")

#ifdef BUILD_CUDA_MODULE
ENUM_ICP_METHOD_DEVICE(BenchmarkICP,
                       PointToPoint,
                       TransformationEstimationType::PointToPoint,
                       "CUDA:0")
ENUM_ICP_METHOD_DEVICE(BenchmarkICP,
                       PointToPlane,
                       TransformationEstimationType::PointToPlane,
                       "CUDA:0")
ENUM_ICP_METHOD_DEVICE(BenchmarkICP,
                       ColoredICP,
                       TransformationEstimationType::ColoredICP,
                       "CUDA:0")
ENUM_ICP_METHOD_DEVICE(BenchmarkDopplerICP,
                       DopplerICP,
                       TransformationEstimationType::DopplerICP,
                       "CUDA:0")
#endif

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
