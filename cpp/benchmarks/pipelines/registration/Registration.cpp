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

#include "Eigen/Eigen"
#include "open3d/io/PointCloudIO.h"
#include "open3d/pipelines/registration/TransformationEstimation.h"
#include "open3d/utility/Console.h"

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

// Eigen::Matrix4d init_trans = Eigen::Matrix4d::Identity();

namespace open3d {

static std::tuple<geometry::PointCloud, geometry::PointCloud> LoadTPointCloud(
        const std::string& source_filename,
        const std::string& target_filename,
        const double voxel_downsample_factor) {
    geometry::PointCloud source_;
    geometry::PointCloud target_;

    io::ReadPointCloud(source_filename, source_, {"auto", false, false, true});
    io::ReadPointCloud(target_filename, target_, {"auto", false, false, true});

    source_ = *source_.VoxelDownSample(voxel_downsample_factor);
    target_ = *target_.VoxelDownSample(voxel_downsample_factor);

    return std::make_tuple(source_, target_);
}

static void RegistrationICPPointToPlaneLegacy(benchmark::State& state) {
    geometry::PointCloud source;
    geometry::PointCloud target;

    std::tie(source, target) = LoadTPointCloud(source_pointcloud_filename,
                                               target_pointcloud_filename,
                                               voxel_downsampling_factor);

    Eigen::Matrix4d init_trans;
    init_trans << 0.862, 0.011, -0.507, 0.5, -0.139, 0.967, -0.215, 0.7, 0.487,
            0.255, 0.835, -1.4, 0.0, 0.0, 0.0, 1.0;

    auto reg_p2plane = open3d::pipelines::registration::RegistrationICP(
            source, target, max_correspondence_dist, init_trans,
            open3d::pipelines::registration::
                    TransformationEstimationPointToPlane(),
            open3d::pipelines::registration::ICPConvergenceCriteria(
                    relative_fitness, relative_rmse, max_iterations));
    utility::LogInfo(" Max iterations: {}, Max_correspondence_distance : {}",
                     max_iterations, max_correspondence_dist);
    utility::LogInfo(" Fitness: {}  Inlier RMSE: {}", reg_p2plane.fitness_,
                     reg_p2plane.inlier_rmse_);

    for (auto _ : state) {
        auto reg_p2plane = open3d::pipelines::registration::RegistrationICP(
                source, target, max_correspondence_dist, init_trans,
                open3d::pipelines::registration::
                        TransformationEstimationPointToPlane(),
                open3d::pipelines::registration::ICPConvergenceCriteria(
                        relative_fitness, relative_rmse, max_iterations));
        utility::LogInfo(" Fitness: {}  Inlier RMSE: {}", reg_p2plane.fitness_,
                         reg_p2plane.inlier_rmse_);
    }
}

static void RegistrationICPPointToPointLegacy(benchmark::State& state) {
    geometry::PointCloud source;
    geometry::PointCloud target;

    std::tie(source, target) = LoadTPointCloud(source_pointcloud_filename,
                                               target_pointcloud_filename,
                                               voxel_downsampling_factor);

    Eigen::Matrix4d init_trans;
    init_trans << 0.862, 0.011, -0.507, 0.5, -0.139, 0.967, -0.215, 0.7, 0.487,
            0.255, 0.835, -1.4, 0.0, 0.0, 0.0, 1.0;

    auto reg_p2plane = open3d::pipelines::registration::RegistrationICP(
            source, target, max_correspondence_dist, init_trans,
            open3d::pipelines::registration::
                    TransformationEstimationPointToPoint(),
            open3d::pipelines::registration::ICPConvergenceCriteria(
                    relative_fitness, relative_rmse, max_iterations));
    utility::LogInfo(" Max iterations: {}, Max_correspondence_distance : {}",
                     max_iterations, max_correspondence_dist);
    utility::LogInfo(" Fitness: {}  Inlier RMSE: {}", reg_p2plane.fitness_,
                     reg_p2plane.inlier_rmse_);

    for (auto _ : state) {
        auto reg_p2plane = open3d::pipelines::registration::RegistrationICP(
                source, target, max_correspondence_dist, init_trans,
                open3d::pipelines::registration::
                        TransformationEstimationPointToPoint(),
                open3d::pipelines::registration::ICPConvergenceCriteria(
                        relative_fitness, relative_rmse, max_iterations));
        utility::LogInfo(" Fitness: {}  Inlier RMSE: {}", reg_p2plane.fitness_,
                         reg_p2plane.inlier_rmse_);
    }
}

BENCHMARK(RegistrationICPPointToPlaneLegacy)->Unit(benchmark::kMillisecond);

BENCHMARK(RegistrationICPPointToPointLegacy)->Unit(benchmark::kMillisecond);

}  // namespace open3d
