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
static double relative_fitness = 1e-6;
static double relative_rmse = 1e-6;
static int max_iterations = 30;

// NNS parameter.
static double max_correspondence_dist = 0.015;

// Initial transformation guess for registation.
// core::Tensor init_trans = core::Tensor::Eye(4, dtype, device);
static std::vector<float> initial_transform_flat{
        0.862, 0.011, -0.507, 0.5,  -0.139, 0.967, -0.215, 0.7,
        0.487, 0.255, 0.835,  -1.4, 0.0,    0.0,   0.0,    1.0};

#define DISPATCH_TRANFORMATION_TYPE(TRANFORMATION_TYPE, ...)                   \
    [&] {                                                                      \
        if (TRANFORMATION_TYPE ==                                              \
            open3d::t::pipelines::registration::TransformationEstimationType:: \
                    PointToPoint) {                                            \
            using scalar_t = open3d::t::pipelines::registration::              \
                    TransformationEstimationPointToPoint;                      \
            return __VA_ARGS__();                                              \
        } else if (TRANFORMATION_TYPE ==                                       \
                   open3d::t::pipelines::registration::                        \
                           TransformationEstimationType::PointToPlane) {       \
            using scalar_t = open3d::t::pipelines::registration::              \
                    TransformationEstimationPointToPoint;                      \
            return __VA_ARGS__();                                              \
        } else {                                                               \
            utility::LogError("Unsupported data type.");                       \
        }                                                                      \
    }()

namespace open3d {
namespace benchmarks {

namespace tregistration_utility {

std::tuple<t::geometry::PointCloud, t::geometry::PointCloud>
LoadTensorPointCloud(const std::string& source_pointcloud_filename,
                     const std::string& target_pointcloud_filename,
                     const double voxel_downsample_factor,
                     const core::Dtype& dtype,
                     const core::Device& device) {
    t::geometry::PointCloud source_, target_;

    t::io::ReadPointCloud(source_pointcloud_filename, source_,
                          {"auto", false, false, true});
    t::io::ReadPointCloud(target_pointcloud_filename, target_,
                          {"auto", false, false, true});

    // Eliminates the case of impractical values (including negative).
    t::geometry::PointCloud source, target;
    if (voxel_downsample_factor > 0.0001) {
        geometry::PointCloud legacy_s = source_.ToLegacyPointCloud();
        geometry::PointCloud legacy_t = target_.ToLegacyPointCloud();
        // TODO: Use to tensor VoxelDownSample.
        legacy_s = *legacy_s.VoxelDownSample(voxel_downsample_factor);
        legacy_t = *legacy_t.VoxelDownSample(voxel_downsample_factor);

        source = t::geometry::PointCloud::FromLegacyPointCloud(legacy_s);
        target = t::geometry::PointCloud::FromLegacyPointCloud(legacy_t);
    } else {
        source = source_.Clone();
        target = target_.Clone();
    }

    t::geometry::PointCloud source_device(device), target_device(device);

    core::Tensor source_points =
            source.GetPoints().To(device, dtype, /*copy=*/true);
    source_device.SetPoints(source_points);

    core::Tensor target_points =
            target.GetPoints().To(device, dtype, /*copy=*/true);
    core::Tensor target_normals =
            target.GetPointNormals().To(device, dtype, /*copy=*/true);
    target_device.SetPoints(target_points);
    target_device.SetPointNormals(target_normals);

    return std::make_tuple(source_device, target_device);
}

}  // namespace tregistration_utility

static void BenchmarkRegistrationICP(
        benchmark::State& state,
        const core::Device& device,
        const t::pipelines::registration::TransformationEstimationType& type_) {
    core::Dtype dtype = core::Dtype::Float32;

    t::geometry::PointCloud source(device), target(device);

    std::tie(source, target) = tregistration_utility::LoadTensorPointCloud(
            source_pointcloud_filename, target_pointcloud_filename,
            voxel_downsampling_factor, dtype, device);

    core::Tensor init_trans =
            core::Tensor(initial_transform_flat, {4, 4}, dtype, device);

    utility::LogInfo(" PointCloud Size: Source: {}  Target: {}",
                     source.GetPoints().GetShape().ToString(),
                     target.GetPoints().GetShape().ToString());
    utility::LogInfo(" Max iterations: {}, Max_correspondence_distance : {}",
                     max_iterations, max_correspondence_dist);

    DISPATCH_TRANFORMATION_TYPE(type_, [&]() {
        scalar_t estimation;
        auto reg_p2plane = open3d::t::pipelines::registration::RegistrationICP(
                source, target, max_correspondence_dist, init_trans, estimation,
                open3d::t::pipelines::registration::ICPConvergenceCriteria(
                        relative_fitness, relative_rmse, max_iterations));

        utility::LogInfo(" Fitness: {}  Inlier RMSE: {}", reg_p2plane.fitness_,
                         reg_p2plane.inlier_rmse_);

        for (auto _ : state) {
            auto reg_p2plane =
                    open3d::t::pipelines::registration::RegistrationICP(
                            source, target, max_correspondence_dist, init_trans,
                            estimation,
                            open3d::t::pipelines::registration::
                                    ICPConvergenceCriteria(relative_fitness,
                                                           relative_rmse,
                                                           max_iterations));

            utility::LogInfo(" Fitness: {}  Inlier RMSE: {}",
                             reg_p2plane.fitness_, reg_p2plane.inlier_rmse_);
        }
    });
}

BENCHMARK_CAPTURE(
        BenchmarkRegistrationICP,
        PointToPlane / CPU,
        core::Device("CPU:0"),
        t::pipelines::registration::TransformationEstimationType::PointToPlane)
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(
        BenchmarkRegistrationICP,
        PointToPlane / CUDA,
        core::Device("CUDA:0"),
        t::pipelines::registration::TransformationEstimationType::PointToPlane)
        ->Unit(benchmark::kMillisecond);
#endif

BENCHMARK_CAPTURE(
        BenchmarkRegistrationICP,
        PointToPoint / CPU,
        core::Device("CPU:0"),
        t::pipelines::registration::TransformationEstimationType::PointToPoint)
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(
        BenchmarkRegistrationICP,
        PointToPoint / CUDA,
        core::Device("CUDA:0"),
        t::pipelines::registration::TransformationEstimationType::PointToPoint)
        ->Unit(benchmark::kMillisecond);
#endif

static t::pipelines::registration::RegistrationResult
GetRegistrationResultAndCorrespondences(
        const t::geometry::PointCloud& source,
        const t::geometry::PointCloud& target,
        open3d::core::nns::NearestNeighborSearch& target_nns,
        double max_correspondence_distance,
        const core::Tensor& transformation) {
    core::Device device = source.GetDevice();
    core::Dtype dtype = core::Dtype::Float32;
    source.GetPoints().AssertDtype(dtype);
    target.GetPoints().AssertDtype(dtype);
    if (target.GetDevice() != device) {
        utility::LogError(
                "Target Pointcloud device {} != Source Pointcloud's device {}.",
                target.GetDevice().ToString(), device.ToString());
    }
    transformation.AssertShape({4, 4});
    transformation.AssertDtype(dtype);

    core::Tensor transformation_device = transformation.To(device);

    t::pipelines::registration::RegistrationResult result(
            transformation_device);
    if (max_correspondence_distance <= 0.0) {
        return result;
    }

    bool check = target_nns.HybridIndex(max_correspondence_distance);
    if (!check) {
        utility::LogError(
                "[Tensor: EvaluateRegistration: "
                "GetRegistrationResultAndCorrespondences: "
                "NearestNeighborSearch::HybridSearch] "
                "Index is not set.");
    }

    core::Tensor distances;
    std::tie(result.correspondence_set.first, result.correspondence_set.second,
             distances) =
            target_nns.Hybrid1NNSearch(source.GetPoints(),
                                       max_correspondence_distance);

    // Number of good correspondences (C).
    int num_correspondences = result.correspondence_set.first.GetLength();

    // Reduction sum of "distances" for error.
    double squared_error =
            static_cast<double>(distances.Sum({0}).Item<float>());
    result.fitness_ = static_cast<double>(num_correspondences) /
                      static_cast<double>(source.GetPoints().GetLength());
    result.inlier_rmse_ =
            std::sqrt(squared_error / static_cast<double>(num_correspondences));
    result.transformation_ = transformation;

    return result;
}

static void BenchmarkGetRegistrationResultAndCorrespondences(
        benchmark::State& state, const core::Device& device) {
    core::Dtype dtype = core::Dtype::Float32;
    t::geometry::PointCloud source(device), target(device);

    std::tie(source, target) = tregistration_utility::LoadTensorPointCloud(
            source_pointcloud_filename, target_pointcloud_filename,
            voxel_downsampling_factor, dtype, device);

    open3d::core::nns::NearestNeighborSearch target_nns(target.GetPoints());

    core::Tensor init_trans =
            core::Tensor(initial_transform_flat, {4, 4}, dtype, device);

    t::geometry::PointCloud source_transformed = source.Clone();
    source_transformed.Transform(init_trans);

    t::pipelines::registration::RegistrationResult result(init_trans);

    result = GetRegistrationResultAndCorrespondences(
            source_transformed, target, target_nns, max_correspondence_dist,
            init_trans);

    utility::LogInfo(
            " Source points: {}, Target points {}, Good correspondences {} ",
            source_transformed.GetPoints().GetLength(),
            target.GetPoints().GetLength(),
            result.correspondence_set.second.GetLength());
    utility::LogInfo(" Fitness: {}  Inlier RMSE: {}", result.fitness_,
                     result.inlier_rmse_);

    for (auto _ : state) {
        result = GetRegistrationResultAndCorrespondences(
                source_transformed, target, target_nns, max_correspondence_dist,
                init_trans);
    }
}

BENCHMARK_CAPTURE(BenchmarkGetRegistrationResultAndCorrespondences,
                  CPU,
                  core::Device("CPU:0"))
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(BenchmarkGetRegistrationResultAndCorrespondences,
                  CUDA,
                  core::Device("CUDA:0"))
        ->Unit(benchmark::kMillisecond);
#endif

static void BenchmarkRegistrationComputeTransformation(
        benchmark::State& state,
        const core::Device& device,
        const t::pipelines::registration::TransformationEstimationType& type_) {
    core::Dtype dtype = core::Dtype::Float32;
    t::geometry::PointCloud source(device), target(device);

    std::tie(source, target) = tregistration_utility::LoadTensorPointCloud(
            source_pointcloud_filename, target_pointcloud_filename,
            voxel_downsampling_factor, dtype, device);

    t::pipelines::registration::CorrespondenceSet corres;

    open3d::core::nns::NearestNeighborSearch target_nns(target.GetPoints());

    core::Tensor init_trans =
            core::Tensor(initial_transform_flat, {4, 4}, dtype, device);

    t::geometry::PointCloud source_transformed = source.Clone();
    source_transformed.Transform(init_trans);

    t::pipelines::registration::RegistrationResult result(init_trans);

    result = GetRegistrationResultAndCorrespondences(
            source_transformed, target, target_nns, max_correspondence_dist,
            init_trans);

    corres = result.correspondence_set;

    DISPATCH_TRANFORMATION_TYPE(type_, [&]() {
        scalar_t estimation;
        // Warm up.
        core::Tensor transformation =
                estimation.ComputeTransformation(source, target, corres);

        for (auto _ : state) {
            core::Tensor transformation =
                    estimation.ComputeTransformation(source, target, corres);
        }
    });
}

BENCHMARK_CAPTURE(
        BenchmarkRegistrationComputeTransformation,
        PointToPlane / CPU,
        core::Device("CPU:0"),
        t::pipelines::registration::TransformationEstimationType::PointToPlane)
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(
        BenchmarkRegistrationComputeTransformation,
        PointToPlane / CUDA,
        core::Device("CUDA:0"),
        t::pipelines::registration::TransformationEstimationType::PointToPlane)
        ->Unit(benchmark::kMillisecond);
#endif

BENCHMARK_CAPTURE(
        BenchmarkRegistrationComputeTransformation,
        PointToPoint / CPU,
        core::Device("CPU:0"),
        t::pipelines::registration::TransformationEstimationType::PointToPoint)
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(
        BenchmarkRegistrationComputeTransformation,
        PointToPoint / CUDA,
        core::Device("CUDA:0"),
        t::pipelines::registration::TransformationEstimationType::PointToPoint)
        ->Unit(benchmark::kMillisecond);
#endif

}  // namespace benchmarks
}  // namespace open3d