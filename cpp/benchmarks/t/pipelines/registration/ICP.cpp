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

#include <benchmark/benchmark.h>

// Remove after testing.
#include "open3d/Open3D.h"

namespace open3d {

std::tuple<t::geometry::PointCloud, t::geometry::PointCloud> LoadTPointCloud(
        const std::string& source_filename,
        const std::string& target_filename,
        const double voxel_downsample_factor,
        const core::Dtype& dtype,
        const core::Device& device) {
    t::geometry::PointCloud source_(device);
    t::geometry::PointCloud target_(device);

    t::io::ReadPointCloud(source_filename, source_,
                          {"auto", false, false, true});
    t::io::ReadPointCloud(target_filename, target_,
                          {"auto", false, false, true});

    geometry::PointCloud legacy_s = source_.ToLegacyPointCloud();
    geometry::PointCloud legacy_t = target_.ToLegacyPointCloud();

    legacy_s = *legacy_s.VoxelDownSample(voxel_downsample_factor);
    legacy_t = *legacy_t.VoxelDownSample(voxel_downsample_factor);

    t::geometry::PointCloud source =
            t::geometry::PointCloud::FromLegacyPointCloud(legacy_s);

    t::geometry::PointCloud target =
            t::geometry::PointCloud::FromLegacyPointCloud(legacy_t);

    core::Tensor source_points =
            source.GetPoints().To(device, dtype, /*copy=*/true);
    t::geometry::PointCloud source_device(device);
    source_device.SetPoints(source_points);

    core::Tensor target_points =
            target.GetPoints().To(device, dtype, /*copy=*/true);
    core::Tensor target_normals =
            target.GetPointNormals().To(device, dtype, /*copy=*/true);
    t::geometry::PointCloud target_device(device);
    target_device.SetPoints(target_points);
    target_device.SetPointNormals(target_normals);

    return std::make_tuple(source_device, target_device);
}

void BenchmarkICPPointToPlane(benchmark::State& state,
                              const core::Device& device) {
    core::Dtype dtype = core::Dtype::Float32;

    t::geometry::PointCloud source(device);
    t::geometry::PointCloud target(device);

    std::tie(source, target) = LoadTPointCloud(
            TEST_DATA_DIR "/ICP/cloud_bin_0.pcd",
            TEST_DATA_DIR "/ICP/cloud_bin_1.pcd", 0.001, dtype, device);

    // core::Tensor init_trans = core::Tensor::Eye(4, dtype, device);
    core::Tensor init_trans =
            core::Tensor::Init<float>({{0.862, 0.011, -0.507, 0.5},
                                       {-0.139, 0.967, -0.215, 0.7},
                                       {0.487, 0.255, 0.835, -1.4},
                                       {0.0, 0.0, 0.0, 1.0}},
                                      device);

    double max_correspondence_dist = 0.02;

    // ICP ConvergenceCriteria:
    double relative_fitness = 1e-6;
    double relative_rmse = 1e-6;
    int max_iterations = 30;

    auto reg_p2plane = open3d::t::pipelines::registration::RegistrationICP(
            source, target, max_correspondence_dist, init_trans,
            open3d::t::pipelines::registration::
                    TransformationEstimationPointToPlane(),
            open3d::t::pipelines::registration::ICPConvergenceCriteria(
                    relative_fitness, relative_rmse, max_iterations));
    utility::LogInfo(" PointCloud Size: Source: {}  Target: {}",
                     source.GetPoints().GetShape().ToString(),
                     target.GetPoints().GetShape().ToString());
    utility::LogInfo(" Max iterations: {}, Max_correspondence_distance : {}",
                     max_iterations, max_correspondence_dist);
    utility::LogInfo(" Fitness: {}  Inlier RMSE: {}", reg_p2plane.fitness_,
                     reg_p2plane.inlier_rmse_);

    for (auto _ : state) {
        auto reg_p2plane = open3d::t::pipelines::registration::RegistrationICP(
                source, target, max_correspondence_dist, init_trans,
                open3d::t::pipelines::registration::
                        TransformationEstimationPointToPlane(),
                open3d::t::pipelines::registration::ICPConvergenceCriteria(
                        relative_fitness, relative_rmse, max_iterations));
        utility::LogInfo(" Fitness: {}  Inlier RMSE: {}", reg_p2plane.fitness_,
                         reg_p2plane.inlier_rmse_);
    }
}

BENCHMARK_CAPTURE(BenchmarkICPPointToPlane, CPU, core::Device("CPU:0"))
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(BenchmarkICPPointToPlane, CUDA, core::Device("CUDA:0"))
        ->Unit(benchmark::kMillisecond);

}  // namespace open3d
