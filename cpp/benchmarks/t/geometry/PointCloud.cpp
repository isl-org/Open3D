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

#include "open3d/t/geometry/PointCloud.h"

#include <benchmark/benchmark.h>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Tensor.h"
#include "open3d/data/Dataset.h"
#include "open3d/io/PointCloudIO.h"
#include "open3d/t/io/PointCloudIO.h"
#include "open3d/utility/DataManager.h"
#include "open3d/visualization/utility/DrawGeometry.h"

namespace open3d {
namespace t {
namespace geometry {

void FromLegacyPointCloud(benchmark::State& state, const core::Device& device) {
    open3d::geometry::PointCloud legacy_pcd;
    size_t num_points = 1000000;  // 1M
    legacy_pcd.points_ =
            std::vector<Eigen::Vector3d>(num_points, Eigen::Vector3d(0, 0, 0));
    legacy_pcd.colors_ =
            std::vector<Eigen::Vector3d>(num_points, Eigen::Vector3d(0, 0, 0));

    // Warm up.
    t::geometry::PointCloud pcd = t::geometry::PointCloud::FromLegacy(
            legacy_pcd, core::Float32, device);
    (void)pcd;

    for (auto _ : state) {
        t::geometry::PointCloud pcd = t::geometry::PointCloud::FromLegacy(
                legacy_pcd, core::Float32, device);
        core::cuda::Synchronize(device);
    }
}

void ToLegacyPointCloud(benchmark::State& state, const core::Device& device) {
    int64_t num_points = 1000000;  // 1M
    PointCloud pcd(device);
    pcd.SetPointPositions(core::Tensor({num_points, 3}, core::Float32, device));
    pcd.SetPointColors(core::Tensor({num_points, 3}, core::Float32, device));

    // Warm up.
    open3d::geometry::PointCloud legacy_pcd = pcd.ToLegacy();
    (void)legacy_pcd;

    for (auto _ : state) {
        open3d::geometry::PointCloud legacy_pcd = pcd.ToLegacy();
        core::cuda::Synchronize(device);
    }
}

data::PLYPointCloud pointcloud_ply;
static const std::string path = pointcloud_ply.GetPath();

void LegacyVoxelDownSample(benchmark::State& state, float voxel_size) {
    auto pcd = open3d::io::CreatePointCloudFromFile(path);
    for (auto _ : state) {
        pcd->VoxelDownSample(voxel_size);
    }
}

void VoxelDownSample(benchmark::State& state,
                     const core::Device& device,
                     float voxel_size,
                     const core::HashBackendType& backend) {
    t::geometry::PointCloud pcd;
    // t::io::CreatePointCloudFromFile lacks support of remove_inf_points and
    // remove_nan_points
    t::io::ReadPointCloud(path, pcd, {"auto", false, false, false});
    pcd = pcd.To(device);

    // Warp up
    pcd.VoxelDownSample(voxel_size, backend);

    for (auto _ : state) {
        pcd.VoxelDownSample(voxel_size, backend);
        core::cuda::Synchronize(device);
    }
}

void Transform(benchmark::State& state, const core::Device& device) {
    PointCloud pcd;
    t::io::ReadPointCloud(path, pcd, {"auto", false, false, false});
    pcd = pcd.To(device);

    core::Dtype dtype = pcd.GetPointPositions().GetDtype();
    core::Tensor transformation = core::Tensor::Init<double>({{1, 0, 0, 1.0},
                                                              {0, 1, 0, 2.0},
                                                              {0, 0, 1, 3.0},
                                                              {0, 0, 0, 1}},
                                                             device)
                                          .To(dtype);

    // Warm Up.
    PointCloud pcd_transformed = pcd.Transform(transformation);

    for (auto _ : state) {
        pcd_transformed = pcd.Transform(transformation);
        core::cuda::Synchronize(device);
    }
}

void EstimateNormals(benchmark::State& state,
                     const core::Device& device,
                     const core::Dtype& dtype,
                     const double voxel_size,
                     const int max_nn,
                     const utility::optional<double> radius) {
    t::geometry::PointCloud pcd;
    t::io::ReadPointCloud(path, pcd, {"auto", false, false, false});

    pcd = pcd.To(device).VoxelDownSample(voxel_size);
    pcd.SetPointPositions(pcd.GetPointPositions().To(dtype));
    if (pcd.HasPointNormals()) {
        pcd.RemovePointAttr("normals");
    }

    // Warp up
    pcd.EstimateNormals(max_nn, radius);
    for (auto _ : state) {
        pcd.EstimateNormals(max_nn, radius);
    }
}

void LegacyEstimateNormals(
        benchmark::State& state,
        const double voxel_size,
        const open3d::geometry::KDTreeSearchParam& search_param) {
    open3d::geometry::PointCloud pcd;
    open3d::io::ReadPointCloud(path, pcd, {"auto", false, false, false});

    auto pcd_down = pcd.VoxelDownSample(voxel_size);

    // Warp up
    pcd_down->EstimateNormals(search_param, true);

    for (auto _ : state) {
        pcd_down->EstimateNormals(search_param, true);
    }
}

BENCHMARK_CAPTURE(FromLegacyPointCloud, CPU, core::Device("CPU:0"))
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(ToLegacyPointCloud, CPU, core::Device("CPU:0"))
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(FromLegacyPointCloud, CUDA, core::Device("CUDA:0"))
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(ToLegacyPointCloud, CUDA, core::Device("CUDA:0"))
        ->Unit(benchmark::kMillisecond);
#endif

#define ENUM_VOXELSIZE(DEVICE, BACKEND)                                       \
    BENCHMARK_CAPTURE(VoxelDownSample, BACKEND##_0_01, DEVICE, 0.01, BACKEND) \
            ->Unit(benchmark::kMillisecond);                                  \
    BENCHMARK_CAPTURE(VoxelDownSample, BACKEND##_0_02, DEVICE, 0.08, BACKEND) \
            ->Unit(benchmark::kMillisecond);                                  \
    BENCHMARK_CAPTURE(VoxelDownSample, BACKEND##_0_04, DEVICE, 0.04, BACKEND) \
            ->Unit(benchmark::kMillisecond);                                  \
    BENCHMARK_CAPTURE(VoxelDownSample, BACKEND##_0_08, DEVICE, 0.08, BACKEND) \
            ->Unit(benchmark::kMillisecond);                                  \
    BENCHMARK_CAPTURE(VoxelDownSample, BACKEND##_0_16, DEVICE, 0.16, BACKEND) \
            ->Unit(benchmark::kMillisecond);                                  \
    BENCHMARK_CAPTURE(VoxelDownSample, BACKEND##_0_32, DEVICE, 0.32, BACKEND) \
            ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
#define ENUM_VOXELDOWNSAMPLE_BACKEND()                                  \
    ENUM_VOXELSIZE(core::Device("CPU:0"), core::HashBackendType::TBB)   \
    ENUM_VOXELSIZE(core::Device("CUDA:0"), core::HashBackendType::Slab) \
    ENUM_VOXELSIZE(core::Device("CUDA:0"), core::HashBackendType::StdGPU)
#else
#define ENUM_VOXELDOWNSAMPLE_BACKEND() \
    ENUM_VOXELSIZE(core::Device("CPU:0"), core::HashBackendType::TBB)
#endif

BENCHMARK_CAPTURE(LegacyVoxelDownSample, Legacy_0_01, 0.01)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(LegacyVoxelDownSample, Legacy_0_02, 0.02)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(LegacyVoxelDownSample, Legacy_0_04, 0.04)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(LegacyVoxelDownSample, Legacy_0_08, 0.08)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(LegacyVoxelDownSample, Legacy_0_16, 0.16)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(LegacyVoxelDownSample, Legacy_0_32, 0.32)
        ->Unit(benchmark::kMillisecond);
ENUM_VOXELDOWNSAMPLE_BACKEND()

BENCHMARK_CAPTURE(Transform, CPU, core::Device("CPU:0"))
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(Transform, CUDA, core::Device("CUDA:0"))
        ->Unit(benchmark::kMillisecond);
#endif

BENCHMARK_CAPTURE(EstimateNormals,
                  CPU F32 Hybrid[0.02 | 30 | 0.06],
                  core::Device("CPU:0"),
                  core::Float32,
                  0.02,
                  30,
                  0.06)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(EstimateNormals,
                  CPU F64 Hybrid[0.02 | 30 | 0.06],
                  core::Device("CPU:0"),
                  core::Float64,
                  0.02,
                  30,
                  0.06)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(EstimateNormals,
                  CPU F32 KNN[0.02 | 30],
                  core::Device("CPU:0"),
                  core::Float32,
                  0.02,
                  30,
                  utility::nullopt)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(EstimateNormals,
                  CPU F64 KNN[0.02 | 30],
                  core::Device("CPU:0"),
                  core::Float64,
                  0.02,
                  30,
                  utility::nullopt)
        ->Unit(benchmark::kMillisecond);
#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(EstimateNormals,
                  CUDA F32 Hybrid[0.02 | 30 | 0.06],
                  core::Device("CUDA:0"),
                  core::Float32,
                  0.02,
                  30,
                  0.06)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(EstimateNormals,
                  CUDA F64 Hybrid[0.02 | 30 | 0.06],
                  core::Device("CUDA:0"),
                  core::Float64,
                  0.02,
                  30,
                  0.06)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(EstimateNormals,
                  CUDA F32 KNN[0.02 | 30],
                  core::Device("CUDA:0"),
                  core::Float32,
                  0.02,
                  30,
                  utility::nullopt)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(EstimateNormals,
                  CUDA F64 KNN[0.02 | 30],
                  core::Device("CUDA:0"),
                  core::Float64,
                  0.02,
                  30,
                  utility::nullopt)
        ->Unit(benchmark::kMillisecond);
#endif

BENCHMARK_CAPTURE(LegacyEstimateNormals,
                  Legacy Hybrid[0.02 | 30 | 0.06],
                  0.02,
                  open3d::geometry::KDTreeSearchParamHybrid(0.06, 30))
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(LegacyEstimateNormals,
                  Legacy KNN[0.02 | 30],
                  0.02,
                  open3d::geometry::KDTreeSearchParamKNN(30))
        ->Unit(benchmark::kMillisecond);

}  // namespace geometry
}  // namespace t
}  // namespace open3d
