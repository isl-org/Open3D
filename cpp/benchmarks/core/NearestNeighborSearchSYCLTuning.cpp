// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Hyperparameter sweep for the SYCL direct-distance KNN kernel
// (KnnDirect, see cpp/open3d/core/nns/kernel/KnnSearchSYCLImpl.h).

#ifdef BUILD_SYCL_MODULE

#include <benchmark/benchmark.h>

#include <sycl/sycl.hpp>

#include "benchmarks/benchmark_utilities/Rand.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/SYCLContext.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/nns/kernel/KnnSearchSYCLImpl.h"

namespace open3d {
namespace benchmarks {
namespace nns {

// Args (via ->ArgsProduct, read with state.range(i)):
//   0: subgroups_per_wg
//   1: tile_points
//   2: k
template <typename T>
static void NNS_KnnDirectTuneT(benchmark::State& state,
                               int64_t num_points,
                               int64_t num_queries) {
    const int64_t subgroups_per_wg = state.range(0);
    const int64_t tile_points = state.range(1);
    const int knn = static_cast<int>(state.range(2));

    const core::Device device("SYCL:0");
    const core::Dtype dtype = core::Dtype::FromType<T>();
    const core::Tensor points =
            benchmarks::Rand({num_points, 3}, 42, {0.0, 1.0}, dtype, device);
    const core::Tensor queries =
            benchmarks::Rand({num_queries, 3}, 43, {0.0, 1.0}, dtype, device);
    core::Tensor out_dist =
            core::Tensor::Empty({num_queries, knn}, dtype, device);
    core::Tensor out_idx =
            core::Tensor::Empty({num_queries, knn}, core::Int32, device);

    sycl::queue queue =
            core::sy::SYCLContext::GetInstance().GetDefaultQueue(device);

    auto run_once = [&]() {
        core::nns::DispatchKnnDirect<T, int32_t>(
                queue, points.GetDataPtr<T>(), queries.GetDataPtr<T>(),
                /*dim=*/3, num_points, num_queries, knn,
                out_dist.GetDataPtr<T>(), out_idx.GetDataPtr<int32_t>(),
                subgroups_per_wg, tile_points);
        queue.wait_and_throw();
    };

    try {
        // Warm-up: also surfaces configs that exceed device limits (SLM
        // size from tile_points, or max work-group size from
        // subgroups_per_wg * SG) as an exception before timing starts.
        run_once();
    } catch (const sycl::exception& e) {
        state.SkipWithError(e.what());
        return;
    }

    for (auto _ : state) {
        run_once();
    }
}

// BENCHMARK_CAPTURE cannot take a template-id (e.g. `Func<float>`) directly
// -- it needs a plain function -- so wrap each instantiation.
static void NNS_KnnDirectTuneFloat(benchmark::State& state,
                                   int64_t num_points,
                                   int64_t num_queries) {
    NNS_KnnDirectTuneT<float>(state, num_points, num_queries);
}
static void NNS_KnnDirectTuneDouble(benchmark::State& state,
                                    int64_t num_points,
                                    int64_t num_queries) {
    NNS_KnnDirectTuneT<double>(state, num_points, num_queries);
}

// 1M points, 1000 queries (matches the NNS_KnnSearch_1M_k* baseline in
// NearestNeighborSearch.cpp). k=1/8/32 sample the small/mid/max K-buckets
// that the direct path dispatches (kSYCLKnnSmallKMax == 32).
BENCHMARK_CAPTURE(NNS_KnnDirectTuneFloat,
                  SYCL_KnnDirectTune_1M_f32,
                  1000000,
                  1000)
        ->ArgNames({"sg_per_wg", "tile_pts", "k"})
        ->ArgsProduct({{2, 4, 8, 16, 32, 64},
                       {64, 128, 256, 512, 1024, 2048, 4096},
                       {1, 8, 32}})
        ->Unit(benchmark::kMillisecond);

// Double spot-check: SLM per work-group is 2*tile_points*NDIM*sizeof(T), so
// double's budget for a given tile_points is half of float's. Only sweep
// tile_points up to the double-safe range (<=2048, see kKnnDirectTilePoints
// sizing discussion) and subgroups_per_wg up to the k=32 work-group-size
// ceiling, to confirm a shared (float+double) default stays valid for double.
BENCHMARK_CAPTURE(NNS_KnnDirectTuneDouble,
                  SYCL_KnnDirectTune_1M_f64,
                  1000000,
                  1000)
        ->ArgNames({"sg_per_wg", "tile_pts", "k"})
        ->ArgsProduct({{8, 16, 32}, {512, 1024, 2048}, {1, 8, 32}})
        ->Unit(benchmark::kMillisecond);

}  // namespace nns
}  // namespace benchmarks
}  // namespace open3d

#endif  // BUILD_SYCL_MODULE
