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

static const double voxel_downsampling_factor = 0.05;

// ICP ConvergenceCriteria.
static double relative_fitness = 1e-6;
static double relative_rmse = 1e-6;
static int max_iterations = 1;

// NNS parameter.
static double max_correspondence_dist = 0.15;

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
    if (voxel_downsample_factor > 0.001) {
        // TODO: Use t::geometry::PointCloud::VoxelDownSample.
        geometry::PointCloud legacy_s = source.ToLegacyPointCloud();
        geometry::PointCloud legacy_t = target.ToLegacyPointCloud();

        legacy_s = *legacy_s.VoxelDownSample(voxel_downsample_factor);
        legacy_t = *legacy_t.VoxelDownSample(voxel_downsample_factor);

        source = t::geometry::PointCloud::FromLegacyPointCloud(legacy_s);
        target = t::geometry::PointCloud::FromLegacyPointCloud(legacy_t);
    } else {
        utility::LogWarning(
                " VoxelDownsample: Impractical voxel size [< 0.001], skiping "
                "downsampling.");
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
    utility::LogDebug(" PointCloud Size: Source: {}  Target: {}",
                      source.GetPoints().GetShape().ToString(),
                      target.GetPoints().GetShape().ToString());
    utility::LogDebug(" Max iterations: {}, Max_correspondence_distance : {}",
                      max_iterations, max_correspondence_dist);
    utility::LogDebug(" Correspondences: {}, Fitness: {}, Inlier RMSE: {}",
                      reg_result.correspondence_set_.first.GetLength(),
                      reg_result.fitness_, reg_result.inlier_rmse_);
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

}  // namespace benchmarks
}  // namespace open3d
