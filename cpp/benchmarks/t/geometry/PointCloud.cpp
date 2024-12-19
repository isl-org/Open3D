// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/geometry/PointCloud.h"

#include <benchmark/benchmark.h>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Tensor.h"
#include "open3d/data/Dataset.h"
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
                     const std::string& reduction) {
    t::geometry::PointCloud pcd;
    // t::io::CreatePointCloudFromFile lacks support of remove_inf_points and
    // remove_nan_points
    t::io::ReadPointCloud(path, pcd, {"auto", false, false, false});
    pcd = pcd.To(device);

    // Warm up.
    pcd.VoxelDownSample(voxel_size, reduction);

    for (auto _ : state) {
        pcd.VoxelDownSample(voxel_size, reduction);
        core::cuda::Synchronize(device);
    }
}

void LegacyUniformDownSample(benchmark::State& state, size_t k) {
    auto pcd = open3d::io::CreatePointCloudFromFile(path);
    for (auto _ : state) {
        pcd->UniformDownSample(k);
    }
}

void UniformDownSample(benchmark::State& state,
                       const core::Device& device,
                       size_t k) {
    t::geometry::PointCloud pcd;
    // t::io::CreatePointCloudFromFile lacks support of remove_inf_points and
    // remove_nan_points
    t::io::ReadPointCloud(path, pcd, {"auto", false, false, false});
    pcd = pcd.To(device);

    // Warm up.
    pcd.UniformDownSample(k);

    for (auto _ : state) {
        pcd.UniformDownSample(k);
        core::cuda::Synchronize(device);
    }
}

void LegacyTransform(benchmark::State& state, const int no_use) {
    open3d::geometry::PointCloud pcd;
    open3d::io::ReadPointCloud(path, pcd, {"auto", false, false, false});

    Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
    transformation.block<3, 1>(0, 3) = Eigen::Vector3d(1, 2, 3);

    // Warm Up.
    open3d::geometry::PointCloud pcd_transformed =
            pcd.Transform(transformation);

    for (auto _ : state) {
        pcd_transformed = pcd.Transform(transformation);
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

void SelectByIndex(benchmark::State& state,
                   bool remove_duplicates,
                   const core::Device& device) {
    PointCloud pcd;
    t::io::ReadPointCloud(path, pcd, {"auto", false, false, false});
    pcd = pcd.To(device);

    const int64_t num_points = pcd.GetPointPositions().GetLength();
    core::Tensor indices =
            core::Tensor::Arange(0, num_points, 1, core::Int64, device);

    // Warm Up.
    PointCloud pcd_selected =
            pcd.SelectByIndex(indices, false, remove_duplicates);

    for (auto _ : state) {
        pcd_selected = pcd.SelectByIndex(indices);
        core::cuda::Synchronize(device);
    }
}

void LegacySelectByIndex(benchmark::State& state, const int no_use) {
    open3d::geometry::PointCloud pcd;
    open3d::io::ReadPointCloud(path, pcd, {"auto", false, false, false});

    const size_t num_points = pcd.points_.size();
    std::vector<size_t> indices(num_points);
    std::iota(indices.begin(), indices.end(), 0);

    // Warm Up.
    open3d::geometry::PointCloud pcd_selected = *pcd.SelectByIndex(indices);

    for (auto _ : state) {
        pcd_selected = *pcd.SelectByIndex(indices);
    }
}

void EstimateNormals(benchmark::State& state,
                     const core::Device& device,
                     const core::Dtype& dtype,
                     const double voxel_size,
                     const utility::optional<int> max_nn,
                     const utility::optional<double> radius) {
    t::geometry::PointCloud pcd;
    t::io::ReadPointCloud(path, pcd, {"auto", false, false, false});

    pcd = pcd.To(device).VoxelDownSample(voxel_size);
    pcd.SetPointPositions(pcd.GetPointPositions().To(dtype));
    if (pcd.HasPointNormals()) {
        pcd.RemovePointAttr("normals");
    }

    // Warm up.
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

    // Warm up.
    pcd_down->EstimateNormals(search_param, true);

    for (auto _ : state) {
        pcd_down->EstimateNormals(search_param, true);
    }
}

void RemoveRadiusOutliers(benchmark::State& state,
                          const core::Device& device,
                          const int nb_points,
                          const double search_radius) {
    t::geometry::PointCloud pcd;
    t::io::ReadPointCloud(path, pcd, {"auto", false, false, false});

    pcd = pcd.To(device).VoxelDownSample(0.01);

    // Warm up.
    pcd.RemoveRadiusOutliers(nb_points, search_radius);
    for (auto _ : state) {
        pcd.RemoveRadiusOutliers(nb_points, search_radius);
    }
}

void RemoveStatisticalOutliers(benchmark::State& state,
                               const core::Device& device,
                               const int nb_neighbors) {
    t::geometry::PointCloud pcd;
    t::io::ReadPointCloud(path, pcd, {"auto", false, false, false});

    pcd = pcd.To(device).VoxelDownSample(0.01);

    // Warm up.
    pcd.RemoveStatisticalOutliers(nb_neighbors, 1.0);
    for (auto _ : state) {
        pcd.RemoveStatisticalOutliers(nb_neighbors, 1.0);
    }
}

void ComputeBoundaryPoints(benchmark::State& state,
                           const core::Device& device,
                           const core::Dtype& dtype,
                           const int max_nn,
                           const double radius) {
    data::DemoCropPointCloud pointcloud_ply;
    const std::string path = pointcloud_ply.GetPointCloudPath();
    t::geometry::PointCloud pcd;
    t::io::ReadPointCloud(path, pcd, {"auto", false, false, false});
    pcd = pcd.To(device);
    pcd.SetPointPositions(pcd.GetPointPositions().To(dtype));
    pcd.SetPointNormals(pcd.GetPointNormals().To(dtype));

    // Warm up.
    pcd.ComputeBoundaryPoints(radius, max_nn);

    for (auto _ : state) {
        pcd.ComputeBoundaryPoints(radius, max_nn);
    }
}

void LegacyRemoveStatisticalOutliers(benchmark::State& state,
                                     const int nb_neighbors) {
    open3d::geometry::PointCloud pcd;
    open3d::io::ReadPointCloud(path, pcd, {"auto", false, false, false});

    auto pcd_down = pcd.VoxelDownSample(0.01);

    // Warm up.
    pcd_down->RemoveStatisticalOutliers(nb_neighbors, 1.0);

    for (auto _ : state) {
        pcd_down->RemoveStatisticalOutliers(nb_neighbors, 1.0);
    }
}

void LegacyRemoveRadiusOutliers(benchmark::State& state,
                                const int nb_points,
                                const double search_radius) {
    open3d::geometry::PointCloud pcd;
    open3d::io::ReadPointCloud(path, pcd, {"auto", false, false, false});

    auto pcd_down = pcd.VoxelDownSample(0.01);

    // Warm up.
    pcd_down->RemoveRadiusOutliers(nb_points, search_radius);

    for (auto _ : state) {
        pcd_down->RemoveRadiusOutliers(nb_points, search_radius);
    }
}

void CropByAxisAlignedBox(benchmark::State& state, const core::Device& device) {
    t::geometry::PointCloud pcd;
    t::io::ReadPointCloud(path, pcd, {"auto", false, false, false});

    pcd = pcd.To(device);
    t::geometry::AxisAlignedBoundingBox box(
            core::Tensor::Init<float>({0, 0, 0}, device),
            core::Tensor::Init<float>({1, 1, 1}, device));

    // Warm up.
    pcd.Crop(box);

    for (auto _ : state) {
        pcd.Crop(box);
    }
}

void CropByOrientedBox(benchmark::State& state, const core::Device& device) {
    t::geometry::PointCloud pcd;
    t::io::ReadPointCloud(path, pcd, {"auto", false, false, false});

    pcd = pcd.To(device);
    t::geometry::OrientedBoundingBox box(
            core::Tensor::Init<float>({0, 0, 0}, device),
            core::Tensor::Eye(3, core::Float32, device),
            core::Tensor::Init<float>({1, 1, 1}, device));

    // Warm up.
    pcd.Crop(box);

    for (auto _ : state) {
        pcd.Crop(box);
    }
}

void LegacyCropByAxisAlignedBox(benchmark::State& state, const int no_use) {
    open3d::geometry::PointCloud pcd;
    open3d::io::ReadPointCloud(path, pcd, {"auto", false, false, false});

    open3d::geometry::AxisAlignedBoundingBox box(Eigen::Vector3d(0, 0, 0),
                                                 Eigen::Vector3d(1, 1, 1));

    // Warm up.
    pcd.Crop(box);

    for (auto _ : state) {
        pcd.Crop(box);
    }
}

void LegacyCropByOrientedBox(benchmark::State& state, const int no_use) {
    open3d::geometry::PointCloud pcd;
    open3d::io::ReadPointCloud(path, pcd, {"auto", false, false, false});

    open3d::geometry::OrientedBoundingBox box(Eigen::Vector3d(0, 0, 0),
                                              Eigen::Matrix3d::Identity(),
                                              Eigen::Vector3d(1, 1, 1));

    // Warm up.
    pcd.Crop(box);

    for (auto _ : state) {
        pcd.Crop(box);
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

#define ENUM_VOXELSIZE(DEVICE, REDUCTION)                              \
    BENCHMARK_CAPTURE(VoxelDownSample, REDUCTION##_0_01, DEVICE, 0.01, \
                      REDUCTION)                                       \
            ->Unit(benchmark::kMillisecond);                           \
    BENCHMARK_CAPTURE(VoxelDownSample, REDUCTION##_0_02, DEVICE, 0.08, \
                      REDUCTION)                                       \
            ->Unit(benchmark::kMillisecond);                           \
    BENCHMARK_CAPTURE(VoxelDownSample, REDUCTION##_0_04, DEVICE, 0.04, \
                      REDUCTION)                                       \
            ->Unit(benchmark::kMillisecond);                           \
    BENCHMARK_CAPTURE(VoxelDownSample, REDUCTION##_0_08, DEVICE, 0.08, \
                      REDUCTION)                                       \
            ->Unit(benchmark::kMillisecond);                           \
    BENCHMARK_CAPTURE(VoxelDownSample, REDUCTION##_0_16, DEVICE, 0.16, \
                      REDUCTION)                                       \
            ->Unit(benchmark::kMillisecond);                           \
    BENCHMARK_CAPTURE(VoxelDownSample, REDUCTION##_0_32, DEVICE, 0.32, \
                      REDUCTION)                                       \
            ->Unit(benchmark::kMillisecond);

const std::string kReductionMean = "mean";
#ifdef BUILD_CUDA_MODULE
#define ENUM_VOXELDOWNSAMPLE_REDUCTION()                  \
    ENUM_VOXELSIZE(core::Device("CPU:0"), kReductionMean) \
    ENUM_VOXELSIZE(core::Device("CUDA:0"), kReductionMean)
#else
#define ENUM_VOXELDOWNSAMPLE_REDUCTION() \
    ENUM_VOXELSIZE(core::Device("CPU:0"), kReductionMean)
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
ENUM_VOXELDOWNSAMPLE_REDUCTION()

BENCHMARK_CAPTURE(LegacyUniformDownSample, Legacy_2, 2)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(LegacyUniformDownSample, Legacy_5, 5)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(LegacyUniformDownSample, Legacy_10, 10)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(UniformDownSample, CPU_2, core::Device("CPU:0"), 2)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(UniformDownSample, CPU_5, core::Device("CPU:0"), 5)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(UniformDownSample, CPU_10, core::Device("CPU:0"), 10)
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(Transform, CPU, core::Device("CPU:0"))
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(SelectByIndex, CPU, false, core::Device("CPU:0"))
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(SelectByIndex,
                  CPU(remove duplicates),
                  true,
                  core::Device("CPU:0"))
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(UniformDownSample, CUDA_2, core::Device("CUDA:0"), 2)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(UniformDownSample, CUDA_5, core::Device("CUDA:0"), 5)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(UniformDownSample, CUDA_10, core::Device("CUDA:0"), 10)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(Transform, CUDA, core::Device("CUDA:0"))
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(SelectByIndex, CUDA, false, core::Device("CUDA:0"))
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(SelectByIndex,
                  CUDA(remove duplicates),
                  true,
                  core::Device("CUDA:0"))
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
BENCHMARK_CAPTURE(EstimateNormals,
                  CPU F32 Radius[0.02 | 0.06],
                  core::Device("CPU:0"),
                  core::Float32,
                  0.02,
                  utility::nullopt,
                  0.06)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(EstimateNormals,
                  CPU F64 Radius[0.02 | 0.06],
                  core::Device("CPU:0"),
                  core::Float64,
                  0.02,
                  utility::nullopt,
                  0.06)
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
BENCHMARK_CAPTURE(EstimateNormals,
                  CUDA F32 Radius[0.02 | 0.06],
                  core::Device("CUDA:0"),
                  core::Float32,
                  0.02,
                  utility::nullopt,
                  0.06)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(EstimateNormals,
                  CUDA F64 Radius[0.02 | 0.06],
                  core::Device("CUDA:0"),
                  core::Float64,
                  0.02,
                  utility::nullopt,
                  0.06)
        ->Unit(benchmark::kMillisecond);
#endif

BENCHMARK_CAPTURE(LegacyTransform, CPU, 1)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(LegacySelectByIndex, CPU, 1)->Unit(benchmark::kMillisecond);

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

BENCHMARK_CAPTURE(LegacyEstimateNormals,
                  Legacy Radius[0.02 | 0.06],
                  0.02,
                  open3d::geometry::KDTreeSearchParamRadius(0.06))
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(
        RemoveRadiusOutliers, CPU[50 | 0.05], core::Device("CPU:0"), 50, 0.03)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(RemoveStatisticalOutliers, CPU[30], core::Device("CPU:0"), 30)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(CropByAxisAlignedBox, CPU, core::Device("CPU:0"))
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(CropByOrientedBox, CPU, core::Device("CPU:0"))
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(ComputeBoundaryPoints,
                  CPU Float32,
                  core::Device("CPU:0"),
                  core::Float32,
                  30,
                  0.02)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(ComputeBoundaryPoints,
                  CPU Float64,
                  core::Device("CPU:0"),
                  core::Float64,
                  30,
                  0.02)
        ->Unit(benchmark::kMillisecond);
#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(
        RemoveRadiusOutliers, CUDA[50 | 0.05], core::Device("CUDA:0"), 50, 0.03)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(CropByAxisAlignedBox, CUDA, core::Device("CUDA:0"))
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(CropByOrientedBox, CUDA, core::Device("CUDA:0"))
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(RemoveStatisticalOutliers,
                  CUDA[30],
                  core::Device("CUDA:0"),
                  30)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(ComputeBoundaryPoints,
                  CUDA Float32,
                  core::Device("CUDA:0"),
                  core::Float32,
                  30,
                  0.02)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(ComputeBoundaryPoints,
                  CUDA Float64,
                  core::Device("CUDA:0"),
                  core::Float64,
                  30,
                  0.02)
        ->Unit(benchmark::kMillisecond);
#endif

BENCHMARK_CAPTURE(LegacyRemoveRadiusOutliers, Legacy[50 | 0.05], 50, 0.03)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(LegacyRemoveStatisticalOutliers, Legacy[30], 30)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(LegacyCropByAxisAlignedBox, Legacy, 1)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(LegacyCropByOrientedBox, Legacy, 1)
        ->Unit(benchmark::kMillisecond);

}  // namespace geometry
}  // namespace t
}  // namespace open3d
