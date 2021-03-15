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
static double max_correspondence_dist = 0.03;

// Initial transformation guess for registation.
static std::vector<float> initial_transform_flat{
        0.862, 0.011, -0.507, 0.5,  -0.139, 0.967, -0.215, 0.7,
        0.487, 0.255, 0.835,  -1.4, 0.0,    0.0,   0.0,    1.0};

namespace open3d {
namespace benchmarks {

static std::tuple<t::geometry::PointCloud, t::geometry::PointCloud>
LoadTensorPointCloudFromFile(const std::string& source_pointcloud_filename,
                             const std::string& target_pointcloud_filename,
                             const double voxel_downsample_factor,
                             const core::Dtype& dtype,
                             const core::Device& device) {
    t::geometry::PointCloud source, target;

    t::io::ReadPointCloud(source_pointcloud_filename, source,
                          {"auto", false, false, true});
    t::io::ReadPointCloud(target_pointcloud_filename, target,
                          {"auto", false, false, true});

    // Eliminates the case of impractical values (including negative).
    if (voxel_downsample_factor > 0.0001) {
        geometry::PointCloud legacy_s = source.ToLegacyPointCloud();
        geometry::PointCloud legacy_t = target.ToLegacyPointCloud();
        // TODO: Use to tensor VoxelDownSample.
        legacy_s = *legacy_s.VoxelDownSample(voxel_downsample_factor);
        legacy_t = *legacy_t.VoxelDownSample(voxel_downsample_factor);

        source = t::geometry::PointCloud::FromLegacyPointCloud(legacy_s);
        target = t::geometry::PointCloud::FromLegacyPointCloud(legacy_t);
    }

    t::geometry::PointCloud source_device(device), target_device(device);

    core::Tensor source_points = source.GetPoints().To(device, dtype);
    source_device.SetPoints(source_points);

    core::Tensor target_points = target.GetPoints().To(device, dtype);
    core::Tensor target_normals = target.GetPointNormals().To(device, dtype);
    target_device.SetPoints(target_points);
    target_device.SetPointNormals(target_normals);

    return std::make_tuple(source_device, target_device);
}

static void BenchmarkRegistrationICP(
        benchmark::State& state,
        const core::Device& device,
        const t::pipelines::registration::TransformationEstimationType& type) {
    core::Dtype dtype = core::Dtype::Float32;

    t::geometry::PointCloud source(device), target(device);

    std::tie(source, target) = LoadTensorPointCloudFromFile(
            source_pointcloud_filename, target_pointcloud_filename,
            voxel_downsampling_factor, dtype, device);

    core::Tensor init_trans =
            core::Tensor(initial_transform_flat, {4, 4}, dtype, device);

    t::pipelines::registration::RegistrationResult reg_result(init_trans);

    if (type == t::pipelines::registration::TransformationEstimationType::
                        PointToPlane) {
        t::pipelines::registration::TransformationEstimationPointToPlane
                estimation;
        // Warm up.
        reg_result = t::pipelines::registration::RegistrationICP(
                source, target, max_correspondence_dist, init_trans, estimation,
                open3d::t::pipelines::registration::ICPConvergenceCriteria(
                        relative_fitness, relative_rmse, max_iterations));
        // Benchmarking.
        for (auto _ : state) {
            reg_result = t::pipelines::registration::RegistrationICP(
                    source, target, max_correspondence_dist, init_trans,
                    estimation,
                    t::pipelines::registration::ICPConvergenceCriteria(
                            relative_fitness, relative_rmse, max_iterations));
        }
    } else if (type == t::pipelines::registration::
                               TransformationEstimationType::PointToPoint) {
        t::pipelines::registration::TransformationEstimationPointToPoint
                estimation;
        // Warm up.
        reg_result = t::pipelines::registration::RegistrationICP(
                source, target, max_correspondence_dist, init_trans, estimation,
                t::pipelines::registration::ICPConvergenceCriteria(
                        relative_fitness, relative_rmse, max_iterations));
        // Benchmarking.
        for (auto _ : state) {
            reg_result = t::pipelines::registration::RegistrationICP(
                    source, target, max_correspondence_dist, init_trans,
                    estimation,
                    t::pipelines::registration::ICPConvergenceCriteria(
                            relative_fitness, relative_rmse, max_iterations));
        }
    }
    utility::LogInfo(" PointCloud Size: Source: {}  Target: {}",
                     source.GetPoints().GetShape().ToString(),
                     target.GetPoints().GetShape().ToString());
    utility::LogInfo(" Max iterations: {}, Max_correspondence_distance : {}",
                     max_iterations, max_correspondence_dist);
    utility::LogInfo(" Fitness: {}  Inlier RMSE: {}", reg_result.fitness_,
                     reg_result.inlier_rmse_);
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
        core::nns::NearestNeighborSearch& target_nns,
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
    std::tie(result.correspondence_set_.first,
             result.correspondence_set_.second, distances) =
            target_nns.Hybrid1NNSearch(source.GetPoints(),
                                       max_correspondence_distance);

    // Number of good correspondences (C).
    int num_correspondences = result.correspondence_set_.first.GetLength();

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

    std::tie(source, target) = LoadTensorPointCloudFromFile(
            source_pointcloud_filename, target_pointcloud_filename,
            voxel_downsampling_factor, dtype, device);

    core::nns::NearestNeighborSearch target_nns(target.GetPoints());

    core::Tensor init_trans =
            core::Tensor(initial_transform_flat, {4, 4}, dtype, device);

    t::geometry::PointCloud source_transformed = source.Clone();
    source_transformed = source_transformed.Transform(init_trans);

    t::pipelines::registration::RegistrationResult result(init_trans);

    result = GetRegistrationResultAndCorrespondences(
            source_transformed, target, target_nns, max_correspondence_dist,
            init_trans);

    utility::LogDebug(
            " Source points: {}, Target points {}, Good correspondences {} ",
            source_transformed.GetPoints().GetLength(),
            target.GetPoints().GetLength(),
            result.correspondence_set_.second.GetLength());
    utility::LogDebug(" Fitness: {}  Inlier RMSE: {}", result.fitness_,
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

static std::tuple<t::geometry::PointCloud,
                  t::geometry::PointCloud,
                  t::pipelines::registration::CorrespondenceSet>
ComputeTransformUtility(
        const core::Device& device,
        const bool& compute_with_postICP_correspondences,
        const t::pipelines::registration::TransformationEstimationType& type) {
    core::Dtype dtype = core::Dtype::Float32;
    t::geometry::PointCloud source(device), target(device);

    std::tie(source, target) = LoadTensorPointCloudFromFile(
            source_pointcloud_filename, target_pointcloud_filename,
            voxel_downsampling_factor, dtype, device);

    core::Tensor init_trans =
            core::Tensor(initial_transform_flat, {4, 4}, dtype, device);

    core::Tensor final_transformation;
    if (compute_with_postICP_correspondences) {
        if (type == t::pipelines::registration::TransformationEstimationType::
                            PointToPoint) {
            t::pipelines::registration::TransformationEstimationPointToPoint
                    estimation;
            auto reg_p2point = t::pipelines::registration::RegistrationICP(
                    source, target, max_correspondence_dist, init_trans,
                    estimation,
                    t::pipelines::registration::ICPConvergenceCriteria(
                            relative_fitness, relative_rmse, max_iterations));
            final_transformation = reg_p2point.transformation_.To(device);
        } else if (type == t::pipelines::registration::
                                   TransformationEstimationType::PointToPlane) {
            t::pipelines::registration::TransformationEstimationPointToPlane
                    estimation;
            auto reg_p2plane = t::pipelines::registration::RegistrationICP(
                    source, target, max_correspondence_dist, init_trans,
                    estimation,
                    t::pipelines::registration::ICPConvergenceCriteria(
                            relative_fitness, relative_rmse, max_iterations));
            final_transformation = reg_p2plane.transformation_.To(device);
        }
    } else {
        final_transformation =
                core::Tensor(initial_transform_flat, {4, 4}, dtype, device);
    }

    t::pipelines::registration::CorrespondenceSet corres;
    core::nns::NearestNeighborSearch target_nns(target.GetPoints());

    t::geometry::PointCloud source_transformed = source.Clone();
    source_transformed = source_transformed.Transform(final_transformation);

    t::pipelines::registration::RegistrationResult result(final_transformation);

    result = GetRegistrationResultAndCorrespondences(
            source_transformed, target, target_nns, max_correspondence_dist,
            final_transformation);

    corres.first = result.correspondence_set_.first.To(device);
    corres.second = result.correspondence_set_.second.To(device);

    return std::make_tuple(source_transformed, target, corres);
}

static void BenchmarkRegistrationComputeTransformation(
        benchmark::State& state,
        const core::Device& device,
        const bool& compute_with_postICP_correspondences,
        const t::pipelines::registration::TransformationEstimationType& type) {
    t::geometry::PointCloud source(device), target(device);
    t::pipelines::registration::CorrespondenceSet corres;

    std::tie(source, target, corres) = ComputeTransformUtility(
            device, compute_with_postICP_correspondences, type);

    if (type == t::pipelines::registration::TransformationEstimationType::
                        PointToPoint) {
        t::pipelines::registration::TransformationEstimationPointToPoint
                estimation;
        // Warm up.
        core::Tensor transformation =
                estimation.ComputeTransformation(source, target, corres);

        for (auto _ : state) {
            core::Tensor transformation =
                    estimation.ComputeTransformation(source, target, corres);
        }
    } else if (type == t::pipelines::registration::
                               TransformationEstimationType::PointToPlane) {
        t::pipelines::registration::TransformationEstimationPointToPlane
                estimation;
        // Warm up.
        core::Tensor transformation =
                estimation.ComputeTransformation(source, target, corres);

        for (auto _ : state) {
            core::Tensor transformation =
                    estimation.ComputeTransformation(source, target, corres);
        }
    }
}

BENCHMARK_CAPTURE(
        BenchmarkRegistrationComputeTransformation,
        PointToPlane[Iteration = First] / CPU,
        core::Device("CPU:0"),
        false,
        t::pipelines::registration::TransformationEstimationType::PointToPlane)
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(
        BenchmarkRegistrationComputeTransformation,
        PointToPlane[Iteration = First] / CUDA,
        core::Device("CUDA:0"),
        false,
        t::pipelines::registration::TransformationEstimationType::PointToPlane)
        ->Unit(benchmark::kMillisecond);
#endif

BENCHMARK_CAPTURE(
        BenchmarkRegistrationComputeTransformation,
        PointToPoint[Iteration = First] / CPU,
        core::Device("CPU:0"),
        false,
        t::pipelines::registration::TransformationEstimationType::PointToPoint)
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(
        BenchmarkRegistrationComputeTransformation,
        PointToPoint[Iteration = First] / CUDA,
        core::Device("CUDA:0"),
        false,
        t::pipelines::registration::TransformationEstimationType::PointToPoint)
        ->Unit(benchmark::kMillisecond);
#endif

BENCHMARK_CAPTURE(
        BenchmarkRegistrationComputeTransformation,
        PointToPlane[Iteration = Last] / CPU,
        core::Device("CPU:0"),
        true,
        t::pipelines::registration::TransformationEstimationType::PointToPlane)
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(
        BenchmarkRegistrationComputeTransformation,
        PointToPlane[Iteration = Last] / CUDA,
        core::Device("CUDA:0"),
        true,
        t::pipelines::registration::TransformationEstimationType::PointToPlane)
        ->Unit(benchmark::kMillisecond);
#endif

BENCHMARK_CAPTURE(
        BenchmarkRegistrationComputeTransformation,
        PointToPoint[Iteration = Last] / CPU,
        core::Device("CPU:0"),
        true,
        t::pipelines::registration::TransformationEstimationType::PointToPoint)
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(
        BenchmarkRegistrationComputeTransformation,
        PointToPoint[Iteration = Last] / CUDA,
        core::Device("CUDA:0"),
        true,
        t::pipelines::registration::TransformationEstimationType::PointToPoint)
        ->Unit(benchmark::kMillisecond);
#endif

}  // namespace benchmarks
}  // namespace open3d
