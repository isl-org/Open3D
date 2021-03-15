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

#include "open3d/pipelines/registration/Registration.h"

#include <benchmark/benchmark.h>

#include <Eigen/Eigen>

#include "open3d/geometry/KDTreeFlann.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/io/PointCloudIO.h"
#include "open3d/pipelines/registration/TransformationEstimation.h"
#include "open3d/utility/Console.h"

// Testing parameters:
// Filename for pointcloud registration data.
static const std::string source_pointcloud_filename =
        TEST_DATA_DIR "/ICP/cloud_bin_0.pcd";
static const std::string target_pointcloud_filename =
        TEST_DATA_DIR "/ICP/cloud_bin_1.pcd";

static const double voxel_downsampling_factor = 0.01;

// ICP ConvergenceCriteria.
static double relative_fitness = 1e-6;
static double relative_rmse = 1e-6;
static int max_iterations = 30;

// NNS parameter.
static double max_correspondence_distance = 0.03;

namespace open3d {
namespace benchmarks {

static std::tuple<geometry::PointCloud, geometry::PointCloud> LoadPointCloud(
        const std::string& source_filename,
        const std::string& target_filename,
        const double voxel_downsample_factor) {
    geometry::PointCloud source;
    geometry::PointCloud target;

    io::ReadPointCloud(source_filename, source, {"auto", false, false, true});
    io::ReadPointCloud(target_filename, target, {"auto", false, false, true});

    // Eliminates the case of impractical values (including negative).
    if (voxel_downsample_factor > 0.0001) {
        source = *source.VoxelDownSample(voxel_downsample_factor);
        target = *target.VoxelDownSample(voxel_downsample_factor);
    }

    return std::make_tuple(source, target);
}

static void BenchmarkRegistrationICPLegacy(
        benchmark::State& state,
        const pipelines::registration::TransformationEstimationType& type) {
    geometry::PointCloud source;
    geometry::PointCloud target;

    std::tie(source, target) = LoadPointCloud(source_pointcloud_filename,
                                              target_pointcloud_filename,
                                              voxel_downsampling_factor);

    Eigen::Matrix4d init_trans;
    init_trans << 0.862, 0.011, -0.507, 0.5, -0.139, 0.967, -0.215, 0.7, 0.487,
            0.255, 0.835, -1.4, 0.0, 0.0, 0.0, 1.0;

    pipelines::registration::RegistrationResult reg_result(init_trans);

    if (type ==
        pipelines::registration::TransformationEstimationType::PointToPlane) {
        pipelines::registration::TransformationEstimationPointToPlane
                estimation;
        // Warm up.
        reg_result = pipelines::registration::RegistrationICP(
                source, target, max_correspondence_distance, init_trans,
                estimation,
                pipelines::registration::ICPConvergenceCriteria(
                        relative_fitness, relative_rmse, max_iterations));
        // Benchmarking.
        for (auto _ : state) {
            reg_result = pipelines::registration::RegistrationICP(
                    source, target, max_correspondence_distance, init_trans,
                    estimation,
                    pipelines::registration::ICPConvergenceCriteria(
                            relative_fitness, relative_rmse, max_iterations));
        }
    } else if (type == pipelines::registration::TransformationEstimationType::
                               PointToPoint) {
        pipelines::registration::TransformationEstimationPointToPoint
                estimation;
        // Warm up.
        reg_result = pipelines::registration::RegistrationICP(
                source, target, max_correspondence_distance, init_trans,
                estimation,
                pipelines::registration::ICPConvergenceCriteria(
                        relative_fitness, relative_rmse, max_iterations));
        // Benchmarking.
        for (auto _ : state) {
            reg_result = pipelines::registration::RegistrationICP(
                    source, target, max_correspondence_distance, init_trans,
                    estimation,
                    pipelines::registration::ICPConvergenceCriteria(
                            relative_fitness, relative_rmse, max_iterations));
        }
    }

    utility::LogInfo(" Max iterations: {}, Max_correspondence_distance : {}",
                     max_iterations, max_correspondence_distance);
    utility::LogInfo(" Fitness: {}  Inlier RMSE: {}", reg_result.fitness_,
                     reg_result.inlier_rmse_);
}

BENCHMARK_CAPTURE(
        BenchmarkRegistrationICPLegacy,
        PointToPlane / CPU,
        pipelines::registration::TransformationEstimationType::PointToPlane)
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(
        BenchmarkRegistrationICPLegacy,
        PointToPoint / CPU,
        pipelines::registration::TransformationEstimationType::PointToPoint)
        ->Unit(benchmark::kMillisecond);

static pipelines::registration::RegistrationResult
GetRegistrationResultAndCorrespondences(
        const geometry::PointCloud& source,
        const geometry::PointCloud& target,
        const geometry::KDTreeFlann& target_kdtree,
        double max_correspondence_distanceance,
        const Eigen::Matrix4d& transformation) {
    pipelines::registration::RegistrationResult result(transformation);
    if (max_correspondence_distanceance <= 0.0) {
        return result;
    }

    double error2 = 0.0;

#pragma omp parallel
    {
        double error2_private = 0.0;
        pipelines::registration::CorrespondenceSet correspondence_set_private;
#pragma omp for nowait
        for (int i = 0; i < (int)source.points_.size(); i++) {
            std::vector<int> indices(1);
            std::vector<double> dists(1);
            const auto& point = source.points_[i];
            if (target_kdtree.SearchHybrid(point,
                                           max_correspondence_distanceance, 1,
                                           indices, dists) > 0) {
                error2_private += dists[0];
                correspondence_set_private.push_back(
                        Eigen::Vector2i(i, indices[0]));
            }
        }
#pragma omp critical
        {
            for (int i = 0; i < (int)correspondence_set_private.size(); i++) {
                result.correspondence_set_.push_back(
                        correspondence_set_private[i]);
            }
            error2 += error2_private;
        }
    }

    if (result.correspondence_set_.empty()) {
        result.fitness_ = 0.0;
        result.inlier_rmse_ = 0.0;
    } else {
        size_t corres_number = result.correspondence_set_.size();
        result.fitness_ = (double)corres_number / (double)source.points_.size();
        result.inlier_rmse_ = std::sqrt(error2 / (double)corres_number);
    }
    return result;
}

static void BenchmarkRegistrationComputeTransformLegacy(
        benchmark::State& state,
        const pipelines::registration::TransformationEstimationType& type) {
    geometry::PointCloud source;
    geometry::PointCloud target;

    std::tie(source, target) = LoadPointCloud(source_pointcloud_filename,
                                              target_pointcloud_filename,
                                              voxel_downsampling_factor);

    Eigen::Matrix4d init_trans;
    init_trans << 0.862, 0.011, -0.507, 0.5, -0.139, 0.967, -0.215, 0.7, 0.487,
            0.255, 0.835, -1.4, 0.0, 0.0, 0.0, 1.0;

    Eigen::Matrix4d transformation = init_trans;
    geometry::KDTreeFlann kdtree;
    kdtree.SetGeometry(target);
    geometry::PointCloud pcd = source;

    pcd = pcd.Transform(init_trans);

    pipelines::registration::RegistrationResult result;
    result = GetRegistrationResultAndCorrespondences(
            pcd, target, kdtree, max_correspondence_distance, transformation);

    if (type ==
        pipelines::registration::TransformationEstimationType::PointToPoint) {
        pipelines::registration::TransformationEstimationPointToPoint
                estimation;
        // Warm up.
        transformation = estimation.ComputeTransformation(
                source, target, result.correspondence_set_);

        for (auto _ : state) {
            transformation = estimation.ComputeTransformation(
                    source, target, result.correspondence_set_);
        }
    } else if (type == pipelines::registration::TransformationEstimationType::
                               PointToPlane) {
        pipelines::registration::TransformationEstimationPointToPlane
                estimation;
        // Warm up.
        transformation = estimation.ComputeTransformation(
                source, target, result.correspondence_set_);

        for (auto _ : state) {
            transformation = estimation.ComputeTransformation(
                    source, target, result.correspondence_set_);
        }
    }
}

BENCHMARK_CAPTURE(
        BenchmarkRegistrationComputeTransformLegacy,
        PointToPlane[Iteration = First] / CPU,
        pipelines::registration::TransformationEstimationType::PointToPlane)
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(
        BenchmarkRegistrationComputeTransformLegacy,
        PointToPoint[Iteration = First] / CPU,
        pipelines::registration::TransformationEstimationType::PointToPoint)
        ->Unit(benchmark::kMillisecond);

}  // namespace benchmarks
}  // namespace open3d