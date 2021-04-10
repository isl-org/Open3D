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

#include "open3d/t/geometry/PointCloud.h"

#include <benchmark/benchmark.h>

#include "open3d/core/Tensor.h"
#include "open3d/io/PointCloudIO.h"
#include "open3d/t/io/PointCloudIO.h"
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
    t::geometry::PointCloud pcd = t::geometry::PointCloud::FromLegacyPointCloud(
            legacy_pcd, core::Dtype::Float32, device);
    (void)pcd;

    for (auto _ : state) {
        t::geometry::PointCloud pcd =
                t::geometry::PointCloud::FromLegacyPointCloud(
                        legacy_pcd, core::Dtype::Float32, device);
    }
}

void ToLegacyPointCloud(benchmark::State& state, const core::Device& device) {
    int64_t num_points = 1000000;  // 1M
    PointCloud pcd(device);
    pcd.SetPoints(core::Tensor({num_points, 3}, core::Dtype::Float32, device));
    pcd.SetPointColors(
            core::Tensor({num_points, 3}, core::Dtype::Float32, device));

    // Warm up.
    open3d::geometry::PointCloud legacy_pcd = pcd.ToLegacyPointCloud();
    (void)legacy_pcd;

    for (auto _ : state) {
        open3d::geometry::PointCloud legacy_pcd = pcd.ToLegacyPointCloud();
    }
}

static const std::string path = std::string(TEST_DATA_DIR) + "/fragment.ply";

void LegacyVoxelDownSample(benchmark::State& state, float voxel_size) {
    auto pcd = open3d::io::CreatePointCloudFromFile(path);
    for (auto _ : state) {
        pcd->VoxelDownSample(voxel_size);
    }
}

void VoxelDownSample(benchmark::State& state,
                     const core::Device& device,
                     float voxel_size,
                     const core::HashmapBackend& backend) {
    t::geometry::PointCloud pcd;
    // t::io::CreatePointCloudFromFile lacks support of remove_inf_points and
    // remove_nan_points
    t::io::ReadPointCloud(path, pcd, {"auto", false, false, false});
    pcd = pcd.To(device);

    // Warp up
    pcd.VoxelDownSample(voxel_size, backend);

    for (auto _ : state) {
        pcd.VoxelDownSample(voxel_size, backend);
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
#define ENUM_VOXELDOWNSAMPLE_BACKEND()                                 \
    ENUM_VOXELSIZE(core::Device("CPU:0"), core::HashmapBackend::TBB)   \
    ENUM_VOXELSIZE(core::Device("CUDA:0"), core::HashmapBackend::Slab) \
    ENUM_VOXELSIZE(core::Device("CUDA:0"), core::HashmapBackend::StdGPU)
#else
#define ENUM_VOXELDOWNSAMPLE_BACKEND() \
    ENUM_VOXELSIZE(core::Device("CPU:0"), core::HashmapBackend::TBB)
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

}  // namespace geometry
}  // namespace t
}  // namespace open3d
