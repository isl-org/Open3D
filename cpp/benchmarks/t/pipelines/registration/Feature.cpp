// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/pipelines/registration/Feature.h"

#include <benchmark/benchmark.h>

#include "open3d/core/CUDAUtils.h"
#include "open3d/data/Dataset.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/pipelines/registration/Feature.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/io/PointCloudIO.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace registration {

data::BunnyMesh pointcloud_ply;
static const std::string path = pointcloud_ply.GetPath();

void LegacyComputeFPFHFeature(benchmark::State& state,
                              utility::optional<int> max_nn,
                              utility::optional<double> radius) {
    auto pcd = open3d::io::CreatePointCloudFromFile(path)->UniformDownSample(3);
    pcd->EstimateNormals();
    for (auto _ : state) {
        if (max_nn.has_value() && radius.has_value()) {
            auto fpfh = open3d::pipelines::registration::ComputeFPFHFeature(
                    *pcd, open3d::geometry::KDTreeSearchParamHybrid(
                                  radius.value(), max_nn.value()));
        } else if (max_nn.has_value() && !radius.has_value()) {
            auto fpfh = open3d::pipelines::registration::ComputeFPFHFeature(
                    *pcd,
                    open3d::geometry::KDTreeSearchParamKNN(max_nn.value()));
        } else if (!max_nn.has_value() && radius.has_value()) {
            auto fpfh = open3d::pipelines::registration::ComputeFPFHFeature(
                    *pcd,
                    open3d::geometry::KDTreeSearchParamRadius(radius.value()));
        }
    }
}

void ComputeFPFHFeature(benchmark::State& state,
                        const core::Device& device,
                        const core::Dtype& dtype,
                        utility::optional<int> max_nn,
                        utility::optional<double> radius) {
    t::geometry::PointCloud pcd;
    t::io::ReadPointCloud(path, pcd);
    pcd = pcd.To(device).UniformDownSample(3);
    pcd.SetPointPositions(pcd.GetPointPositions().To(dtype));
    pcd.EstimateNormals();

    core::Tensor fpfh;
    // Warm up.
    fpfh = t::pipelines::registration::ComputeFPFHFeature(pcd, max_nn, radius);

    for (auto _ : state) {
        fpfh = t::pipelines::registration::ComputeFPFHFeature(pcd, max_nn,
                                                              radius);
        core::cuda::Synchronize(device);
    }
}

BENCHMARK_CAPTURE(LegacyComputeFPFHFeature,
                  Legacy Hybrid[0.01 | 100],
                  100,
                  0.01)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(LegacyComputeFPFHFeature, Legacy Hybrid[0.02 | 50], 50, 0.02)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(LegacyComputeFPFHFeature,
                  Legacy Hybrid[0.02 | 100],
                  100,
                  0.02)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(LegacyComputeFPFHFeature,
                  Legacy KNN[50],
                  50,
                  utility::nullopt)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(LegacyComputeFPFHFeature,
                  Legacy KNN[100],
                  100,
                  utility::nullopt)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(LegacyComputeFPFHFeature,
                  Legacy Radius[0.01],
                  utility::nullopt,
                  0.01)
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(LegacyComputeFPFHFeature,
                  Legacy Radius[0.02],
                  utility::nullopt,
                  0.02)
        ->Unit(benchmark::kMillisecond);

#define ENUM_FPFH_METHOD_DEVICE(METHOD_NAME, MAX_NN, RADIUS, DEVICE)       \
    BENCHMARK_CAPTURE(ComputeFPFHFeature, METHOD_NAME##_Float32,           \
                      core::Device(DEVICE), core::Float32, MAX_NN, RADIUS) \
            ->Unit(benchmark::kMillisecond);                               \
    BENCHMARK_CAPTURE(ComputeFPFHFeature, DEVICE METHOD_NAME##_Float64,    \
                      core::Device(DEVICE), core::Float32, MAX_NN, RADIUS) \
            ->Unit(benchmark::kMillisecond);

ENUM_FPFH_METHOD_DEVICE(CPU[0.02 | 50] Hybrid, 100, 0.01, "CPU:0")
ENUM_FPFH_METHOD_DEVICE(CPU[0.02 | 50] Hybrid, 50, 0.02, "CPU:0")
ENUM_FPFH_METHOD_DEVICE(CPU[0.02 | 100] Hybrid, 100, 0.02, "CPU:0")
ENUM_FPFH_METHOD_DEVICE(CPU[50] KNN, 50, utility::nullopt, "CPU:0")
ENUM_FPFH_METHOD_DEVICE(CPU[100] KNN, 100, utility::nullopt, "CPU:0")
ENUM_FPFH_METHOD_DEVICE(CPU[0.01] Radius, utility::nullopt, 0.01, "CPU:0")
ENUM_FPFH_METHOD_DEVICE(CPU[0.02] Radius, utility::nullopt, 0.02, "CPU:0")

#ifdef BUILD_CUDA_MODULE
ENUM_FPFH_METHOD_DEVICE(CUDA[0.02 | 50] Hybrid, 100, 0.01, "CUDA:0")
ENUM_FPFH_METHOD_DEVICE(CUDA[0.02 | 50] Hybrid, 50, 0.01, "CUDA:0")
ENUM_FPFH_METHOD_DEVICE(CUDA[0.02 | 100] Hybrid, 100, 0.02, "CUDA:0")
ENUM_FPFH_METHOD_DEVICE(CUDA[50] KNN, 50, utility::nullopt, "CUDA:0")
ENUM_FPFH_METHOD_DEVICE(CUDA[100] KNN, 100, utility::nullopt, "CUDA:0")
ENUM_FPFH_METHOD_DEVICE(CUDA[0.01] Radius, utility::nullopt, 0.01, "CUDA:0")
ENUM_FPFH_METHOD_DEVICE(CUDA[0.02] Radius, utility::nullopt, 0.02, "CUDA:0")
#endif

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
