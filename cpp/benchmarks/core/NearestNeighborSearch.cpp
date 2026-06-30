// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/nns/NearestNeighborSearch.h"

#include <benchmark/benchmark.h>

#include <cmath>

#include "benchmarks/benchmark_utilities/Rand.h"
#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/Tensor.h"

namespace open3d {
namespace benchmarks {
namespace nns {

static void DeviceSync(const core::Device& device) {
    core::cuda::Synchronize(device);
}

/// Ball radius in [0,1]^3 with roughly k points expected inside (uniform).
static double RadiusForExpectedK(int64_t num_points, int k) {
    constexpr double kPi = 3.14159265358979323846;
    if (num_points <= 0 || k <= 0) {
        return 0.1;
    }
    const double frac =
            static_cast<double>(k) / static_cast<double>(num_points);
    return std::cbrt((3.0 * frac) / (4.0 * kPi));
}

void NNS_KnnBuild(benchmark::State& state,
                  int64_t num_points,
                  const core::Device& device) {
    const core::Tensor points = benchmarks::Rand(
            {num_points, 3}, 42, {0.0, 1.0}, core::Float32, device);
    {
        core::nns::NearestNeighborSearch nns(points);
        nns.KnnIndex();
        DeviceSync(device);
    }

    for (auto _ : state) {
        core::nns::NearestNeighborSearch nns(points);
        nns.KnnIndex();
        DeviceSync(device);
    }
}

void NNS_KnnSearch(benchmark::State& state,
                   int64_t num_points,
                   int64_t num_queries,
                   int knn,
                   const core::Device& device) {
    const core::Tensor points = benchmarks::Rand(
            {num_points, 3}, 42, {0.0, 1.0}, core::Float32, device);
    const core::Tensor queries = benchmarks::Rand(
            {num_queries, 3}, 43, {0.0, 1.0}, core::Float32, device);
    core::nns::NearestNeighborSearch nns(points);
    nns.KnnIndex();
    {
        auto result = nns.KnnSearch(queries, knn);
        (void)result;
        DeviceSync(device);
    }

    for (auto _ : state) {
        auto result = nns.KnnSearch(queries, knn);
        (void)result;
        DeviceSync(device);
    }
}

void NNS_FrsBuild(benchmark::State& state,
                  int64_t num_points,
                  int knn_hint,
                  const core::Device& device) {
    const core::Tensor points = benchmarks::Rand(
            {num_points, 3}, 44, {0.0, 1.0}, core::Float32, device);
    const double radius = RadiusForExpectedK(num_points, knn_hint);
    {
        core::nns::NearestNeighborSearch nns(points);
        nns.FixedRadiusIndex(radius);
        DeviceSync(device);
    }

    for (auto _ : state) {
        core::nns::NearestNeighborSearch nns(points);
        nns.FixedRadiusIndex(radius);
        DeviceSync(device);
    }
}

void NNS_FrsSearch(benchmark::State& state,
                   int64_t num_points,
                   int64_t num_queries,
                   int knn_hint,
                   const core::Device& device) {
    const core::Tensor points = benchmarks::Rand(
            {num_points, 3}, 45, {0.0, 1.0}, core::Float32, device);
    const core::Tensor queries = benchmarks::Rand(
            {num_queries, 3}, 46, {0.0, 1.0}, core::Float32, device);
    const double radius = RadiusForExpectedK(num_points, knn_hint);
    core::nns::NearestNeighborSearch nns(points);
    nns.FixedRadiusIndex(radius);
    {
        auto result = nns.FixedRadiusSearch(queries, radius);
        (void)result;
        DeviceSync(device);
    }

    for (auto _ : state) {
        auto result = nns.FixedRadiusSearch(queries, radius);
        (void)result;
        DeviceSync(device);
    }
}

void NNS_HybridBuild(benchmark::State& state,
                     int64_t num_points,
                     int max_knn,
                     const core::Device& device) {
    const core::Tensor points = benchmarks::Rand(
            {num_points, 3}, 47, {0.0, 1.0}, core::Float32, device);
    const double radius = RadiusForExpectedK(num_points, max_knn);
    {
        core::nns::NearestNeighborSearch nns(points);
        nns.HybridIndex(radius);
        DeviceSync(device);
    }

    for (auto _ : state) {
        core::nns::NearestNeighborSearch nns(points);
        nns.HybridIndex(radius);
        DeviceSync(device);
    }
}

void NNS_HybridSearch(benchmark::State& state,
                      int64_t num_points,
                      int64_t num_queries,
                      int max_knn,
                      const core::Device& device) {
    const core::Tensor points = benchmarks::Rand(
            {num_points, 3}, 48, {0.0, 1.0}, core::Float32, device);
    const core::Tensor queries = benchmarks::Rand(
            {num_queries, 3}, 49, {0.0, 1.0}, core::Float32, device);
    const double radius = RadiusForExpectedK(num_points, max_knn);
    core::nns::NearestNeighborSearch nns(points);
    nns.HybridIndex(radius);
    {
        auto result = nns.HybridSearch(queries, radius, max_knn);
        (void)result;
        DeviceSync(device);
    }

    for (auto _ : state) {
        auto result = nns.HybridSearch(queries, radius, max_knn);
        (void)result;
        DeviceSync(device);
    }
}

#define ENUM_NNS_BUILD(FN, K_OR_KNN, DEVICE, TAG)               \
    BENCHMARK_CAPTURE(FN, TAG##_10k, 10000, K_OR_KNN, DEVICE)   \
            ->Unit(benchmark::kMillisecond);                    \
    BENCHMARK_CAPTURE(FN, TAG##_100k, 100000, K_OR_KNN, DEVICE) \
            ->Unit(benchmark::kMillisecond);                    \
    BENCHMARK_CAPTURE(FN, TAG##_1M, 1000000, K_OR_KNN, DEVICE)  \
            ->Unit(benchmark::kMillisecond);

#define ENUM_NNS_SEARCH(FN, DEVICE, TAG)                            \
    BENCHMARK_CAPTURE(FN, TAG##_10k_k1, 10000, 1000, 1, DEVICE)     \
            ->Unit(benchmark::kMillisecond);                        \
    BENCHMARK_CAPTURE(FN, TAG##_10k_k4, 10000, 1000, 4, DEVICE)     \
            ->Unit(benchmark::kMillisecond);                        \
    BENCHMARK_CAPTURE(FN, TAG##_10k_k8, 10000, 1000, 8, DEVICE)     \
            ->Unit(benchmark::kMillisecond);                        \
    BENCHMARK_CAPTURE(FN, TAG##_10k_k32, 10000, 1000, 32, DEVICE)   \
            ->Unit(benchmark::kMillisecond);                        \
    BENCHMARK_CAPTURE(FN, TAG##_100k_k1, 100000, 1000, 1, DEVICE)   \
            ->Unit(benchmark::kMillisecond);                        \
    BENCHMARK_CAPTURE(FN, TAG##_100k_k4, 100000, 1000, 4, DEVICE)   \
            ->Unit(benchmark::kMillisecond);                        \
    BENCHMARK_CAPTURE(FN, TAG##_100k_k8, 100000, 1000, 8, DEVICE)   \
            ->Unit(benchmark::kMillisecond);                        \
    BENCHMARK_CAPTURE(FN, TAG##_100k_k32, 100000, 1000, 32, DEVICE) \
            ->Unit(benchmark::kMillisecond);                        \
    BENCHMARK_CAPTURE(FN, TAG##_1M_k1, 1000000, 1000, 1, DEVICE)    \
            ->Unit(benchmark::kMillisecond);                        \
    BENCHMARK_CAPTURE(FN, TAG##_1M_k4, 1000000, 1000, 4, DEVICE)    \
            ->Unit(benchmark::kMillisecond);                        \
    BENCHMARK_CAPTURE(FN, TAG##_1M_k8, 1000000, 1000, 8, DEVICE)    \
            ->Unit(benchmark::kMillisecond);                        \
    BENCHMARK_CAPTURE(FN, TAG##_1M_k32, 1000000, 1000, 32, DEVICE)  \
            ->Unit(benchmark::kMillisecond);

#define ENUM_NNS_DEVICE(DEVICE, TAG)                                     \
    BENCHMARK_CAPTURE(NNS_KnnBuild, TAG##_KnnBuild, 10000, DEVICE)       \
            ->Unit(benchmark::kMillisecond);                             \
    BENCHMARK_CAPTURE(NNS_KnnBuild, TAG##_KnnBuild_100k, 100000, DEVICE) \
            ->Unit(benchmark::kMillisecond);                             \
    BENCHMARK_CAPTURE(NNS_KnnBuild, TAG##_KnnBuild_1M, 1000000, DEVICE)  \
            ->Unit(benchmark::kMillisecond);                             \
    ENUM_NNS_SEARCH(NNS_KnnSearch, DEVICE, TAG##_KnnSearch)              \
    ENUM_NNS_BUILD(NNS_FrsBuild, 8, DEVICE, TAG##_FrsBuild_k8)           \
    ENUM_NNS_SEARCH(NNS_FrsSearch, DEVICE, TAG##_FrsSearch)              \
    ENUM_NNS_BUILD(NNS_HybridBuild, 8, DEVICE, TAG##_HybridBuild_k8)     \
    ENUM_NNS_SEARCH(NNS_HybridSearch, DEVICE, TAG##_HybridSearch)

#ifdef BUILD_CUDA_MODULE
#define ENUM_NNS_CUDA() ENUM_NNS_DEVICE(core::Device("CUDA:0"), CUDA)
#else
#define ENUM_NNS_CUDA()
#endif

#ifdef BUILD_SYCL_MODULE
#define ENUM_NNS_SYCL() ENUM_NNS_DEVICE(core::Device("SYCL:0"), SYCL)
#else
#define ENUM_NNS_SYCL()
#endif

ENUM_NNS_DEVICE(core::Device("CPU:0"), CPU)
ENUM_NNS_CUDA()
ENUM_NNS_SYCL()

}  // namespace nns
}  // namespace benchmarks
}  // namespace open3d
