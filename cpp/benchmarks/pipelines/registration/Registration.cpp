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

#include "open3d/pipelines/registration/Registration.h"

#include <benchmark/benchmark.h>

#include <Eigen/Eigen>

#include "open3d/data/Dataset.h"
#include "open3d/geometry/KDTreeFlann.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/io/PointCloudIO.h"
#include "open3d/pipelines/registration/TransformationEstimation.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace pipelines {
namespace registration {

// Testing parameters:
// ICP ConvergenceCriteria.
static const double relative_fitness = 1e-6;
static const double relative_rmse = 1e-6;
static const int max_iterations = 30;

// NNS parameter.
static const double max_correspondence_distance = 0.05;

static std::tuple<geometry::PointCloud, geometry::PointCloud> LoadPointCloud(
        const std::string& source_filename,
        const std::string& target_filename,
        const double voxel_downsample_factor) {
    geometry::PointCloud source;
    geometry::PointCloud target;

    io::ReadPointCloud(source_filename, source, {"auto", false, false, true});
    io::ReadPointCloud(target_filename, target, {"auto", false, false, true});

    // Eliminates the case of impractical values (including negative).
    if (voxel_downsample_factor > 0.001) {
        source = *source.VoxelDownSample(voxel_downsample_factor);
        target = *target.VoxelDownSample(voxel_downsample_factor);
    } else {
        utility::LogWarning(
                " VoxelDownsample: Impractical voxel size [< 0.001], skipping "
                "downsampling.");
    }

    return std::make_tuple(source, target);
}

static void BenchmarkICPLegacy(benchmark::State& state,
                               const TransformationEstimationType& type) {
    data::DemoICPPointClouds demo_icp_pointclouds;
    geometry::PointCloud source, target;
    std::tie(source, target) = LoadPointCloud(
            demo_icp_pointclouds.GetPaths(0), demo_icp_pointclouds.GetPaths(1),
            /*voxel_downsampling_factor =*/0.02);

    std::shared_ptr<TransformationEstimation> estimation;
    if (type == TransformationEstimationType::PointToPlane) {
        estimation = std::make_shared<TransformationEstimationPointToPlane>();
    } else if (type == TransformationEstimationType::PointToPoint) {
        estimation = std::make_shared<TransformationEstimationPointToPoint>();
    }

    Eigen::Matrix4d init_trans;
    init_trans << 0.862, 0.011, -0.507, 0.5, -0.139, 0.967, -0.215, 0.7, 0.487,
            0.255, 0.835, -1.4, 0.0, 0.0, 0.0, 1.0;

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

    utility::LogDebug(" Max iterations: {}, Max_correspondence_distance : {}",
                      max_iterations, max_correspondence_distance);
    utility::LogDebug(" Fitness: {}  Inlier RMSE: {}", reg_result.fitness_,
                      reg_result.inlier_rmse_);
}

BENCHMARK_CAPTURE(BenchmarkICPLegacy,
                  PointToPlane / CPU,
                  TransformationEstimationType::PointToPlane)
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(BenchmarkICPLegacy,
                  PointToPoint / CPU,
                  TransformationEstimationType::PointToPoint)
        ->Unit(benchmark::kMillisecond);

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
