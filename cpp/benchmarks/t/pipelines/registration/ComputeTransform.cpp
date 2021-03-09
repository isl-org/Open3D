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
#include "open3d/t/pipelines/registration/TransformationEstimation.h"

namespace open3d {

static std::tuple<t::geometry::PointCloud,
                  t::geometry::PointCloud,
                  t::pipelines::registration::CorrespondenceSet>
LoadData(const std::string& source_points_filename,
         const std::string& target_points_filename,
         const std::string& target_normals_filename,
         const std::string& correspondence_first_filename,
         const std::string& correspondence_second_filename,
         const core::Dtype& dtype,
         const core::Device& device) {
    core::Tensor source_points = core::Tensor::Load(source_points_filename);
    core::Tensor target_points = core::Tensor::Load(target_points_filename);
    core::Tensor target_normals = core::Tensor::Load(target_normals_filename);
    core::Tensor corres_first =
            core::Tensor::Load(correspondence_first_filename);
    core::Tensor corres_second =
            core::Tensor::Load(correspondence_second_filename);

    utility::LogInfo(
            " Source points: {}, Target points {}, Good correspondences {} ",
            source_points.GetShape()[0], target_points.GetShape()[0],
            corres_second.GetShape()[0]);

    // Creating CorrespondenceSet from saved testing_data, on device.
    core::Tensor corres_first_device = corres_first.To(device);
    core::Tensor corres_second_device = corres_second.To(device);
    t::pipelines::registration::CorrespondenceSet corres =
            std::make_pair(corres_first_device, corres_second_device);

    // Creating source pointcloud from saved testing_data, on device.
    core::Tensor source_points_ =
            source_points.To(device, dtype, /*copy=*/true);
    t::geometry::PointCloud source_device(device);
    source_device.SetPoints(source_points_);

    // Creating target pointcloud from saved testing_data, on device.
    core::Tensor target_points_ =
            target_points.To(device, dtype, /*copy=*/true);
    core::Tensor target_normals_ =
            target_normals.To(device, dtype, /*copy=*/true);
    t::geometry::PointCloud target_device(device);
    target_device.SetPoints(target_points_);
    target_device.SetPointNormals(target_normals_);

    return std::make_tuple(source_device, target_device, corres);
}

static void RegistrationComputePosePointToPlane(benchmark::State& state,
                                                const core::Device& device) {
    core::Dtype dtype = core::Dtype::Float32;
    t::pipelines::registration::CorrespondenceSet corres;
    t::geometry::PointCloud source_device(device);
    t::geometry::PointCloud target_device(device);

    std::tie(source_device, target_device, corres) = LoadData(
            TEST_DATA_DIR "/ICP/testing_data/source_points_cloud_bin.npy",
            TEST_DATA_DIR "/ICP/testing_data/target_points_cloud_bin.npy",
            TEST_DATA_DIR "/ICP/testing_data/target_normals_cloud_bin.npy",
            TEST_DATA_DIR
            "/ICP/testing_data/correspondence_first_cloud_bin.npy",
            TEST_DATA_DIR
            "/ICP/testing_data/correspondence_second_cloud_bin.npy",
            dtype, device);

    auto estimation = open3d::t::pipelines::registration::
            TransformationEstimationPointToPlane();

    // Warm up.
    core::Tensor transform = estimation.ComputeTransformation(
            source_device, target_device, corres);

    for (auto _ : state) {
        core::Tensor transform = estimation.ComputeTransformation(
                source_device, target_device, corres);
    }
}

static void RegistrationComputePosePointToPoint(benchmark::State& state,
                                                const core::Device& device) {
    core::Dtype dtype = core::Dtype::Float32;
    t::pipelines::registration::CorrespondenceSet corres;
    t::geometry::PointCloud source_device(device);
    t::geometry::PointCloud target_device(device);

    std::tie(source_device, target_device, corres) = LoadData(
            TEST_DATA_DIR "/ICP/testing_data/source_points_cloud_bin.npy",
            TEST_DATA_DIR "/ICP/testing_data/target_points_cloud_bin.npy",
            TEST_DATA_DIR "/ICP/testing_data/target_normals_cloud_bin.npy",
            TEST_DATA_DIR
            "/ICP/testing_data/correspondence_first_cloud_bin.npy",
            TEST_DATA_DIR
            "/ICP/testing_data/correspondence_second_cloud_bin.npy",
            dtype, device);

    auto estimation = open3d::t::pipelines::registration::
            TransformationEstimationPointToPoint();

    // Warm up.
    core::Tensor transform = estimation.ComputeTransformation(
            source_device, target_device, corres);

    for (auto _ : state) {
        core::Tensor transform = estimation.ComputeTransformation(
                source_device, target_device, corres);
    }
}

BENCHMARK_CAPTURE(RegistrationComputePosePointToPlane,
                  CPU,
                  core::Device("CPU:0"))
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(RegistrationComputePosePointToPlane,
                  CUDA,
                  core::Device("CUDA:0"))
        ->Unit(benchmark::kMillisecond);
#endif

BENCHMARK_CAPTURE(RegistrationComputePosePointToPoint,
                  CPU,
                  core::Device("CPU:0"))
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(RegistrationComputePosePointToPoint,
                  CUDA,
                  core::Device("CUDA:0"))
        ->Unit(benchmark::kMillisecond);
#endif

}  // namespace open3d
