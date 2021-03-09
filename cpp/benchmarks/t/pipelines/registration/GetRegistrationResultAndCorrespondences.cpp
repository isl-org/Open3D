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

#include "open3d/core/Tensor.h"
#include "open3d/core/nns/NearestNeighborSearch.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/io/PointCloudIO.h"
#include "open3d/t/pipelines/registration/Registration.h"
#include "open3d/t/pipelines/registration/TransformationEstimation.h"
#include "open3d/utility/Console.h"
#include "open3d/utility/Helper.h"

namespace open3d {

static std::tuple<t::geometry::PointCloud, t::geometry::PointCloud>
LoadTPointCloud(const std::string& source_filename,
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
            target_nns.SqueezedHybridSearch(source.GetPoints(),
                                            max_correspondence_distance);

    // Number of good correspondences (C).
    int num_correspondences = result.correspondence_set.first.GetShape()[0];

    // Reduction sum of "distances" for error.
    double squared_error =
            static_cast<double>(distances.Sum({0}).Item<float>());
    result.fitness_ = static_cast<double>(num_correspondences) /
                      static_cast<double>(source.GetPoints().GetShape()[0]);
    result.inlier_rmse_ =
            std::sqrt(squared_error / static_cast<double>(num_correspondences));
    result.transformation_ = transformation;

    return result;
}

static void BenchmarkGetRegistrationResultAndCorrespondences(
        benchmark::State& state, const core::Device& device) {
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
    open3d::core::nns::NearestNeighborSearch target_nns(target.GetPoints());
    t::geometry::PointCloud source_transformed = source.Clone();
    source_transformed.Transform(init_trans);

    t::pipelines::registration::RegistrationResult result(init_trans);

    result = GetRegistrationResultAndCorrespondences(
            source_transformed, target, target_nns, max_correspondence_dist,
            init_trans);

    utility::LogInfo(
            " Source points: {}, Target points {}, Good correspondences {} ",
            source_transformed.GetPoints().GetShape()[0],
            target.GetPoints().GetShape()[0],
            result.correspondence_set.second.GetShape()[0]);
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
}  // namespace open3d
