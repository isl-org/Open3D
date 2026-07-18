// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Hyperparameter sweep for the SYCL AddMM-tiled KNN path (KnnSearchSYCL's
// fused small/mid-k branch, see cpp/open3d/core/nns/KnnSearchOpsSYCL.cpp and
// ChooseTileSize in cpp/open3d/core/nns/kernel/KnnSearchSYCLImpl.h), and
// a companion sweep to locate the optimal direct-vs-AddMM routing boundary
// (UseKnnDirect in KnnSearchSYCLImpl.h).

#ifdef BUILD_SYCL_MODULE

#include <benchmark/benchmark.h>

#include "benchmarks/benchmark_utilities/Rand.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/nns/KnnIndex.h"

namespace open3d {
namespace benchmarks {
namespace nns {

namespace {

// ── Part 1: AddMM tile-shape tuning ────────────────────────────────────────
// Args (via ->ArgsProduct, read with state.range(i)):
//   0: tile_bytes, in MiB
//   1: max_tile_queries
//   2: tile_points_alignment
// dim/knn are fixed per BENCHMARK_CAPTURE call (captured, not swept) so the
// grid stays small; two captures cover the two ways a (dim, knn) pair can be
// routed to AddMM in production: dim > kKnnDirectMaxDim, or
// knn > kSYCLKnnSmallKMax.
template <typename T>
void NNS_KnnAddMMTuneT(benchmark::State& state,
                       int64_t num_points,
                       int64_t num_queries,
                       int64_t dim,
                       int knn) {
    const int64_t tile_bytes = state.range(0) * 1024 * 1024;
    const int64_t max_tile_queries = state.range(1);
    const int64_t tile_points_alignment = state.range(2);

    const core::Device device("SYCL:0");
    const core::Dtype dtype = core::Dtype::FromType<T>();
    const core::Tensor points =
            benchmarks::Rand({num_points, dim}, 42, {0.0, 1.0}, dtype, device);
    const core::Tensor queries =
            benchmarks::Rand({num_queries, dim}, 43, {0.0, 1.0}, dtype, device);
    const core::Tensor points_row_splits =
            core::Tensor::Init<int64_t>({0, num_points});
    const core::Tensor queries_row_splits =
            core::Tensor::Init<int64_t>({0, num_queries});

    auto run_once = [&]() {
        core::Tensor neighbors_index, neighbors_distance;
        core::Tensor neighbors_row_splits =
                core::Tensor::Empty({num_queries + 1}, core::Int64);
        // force_addmm_path=true: always exercise the AddMM path being
        // tuned, regardless of whether (dim, knn) would also qualify for
        // the direct-distance path.
        core::nns::KnnSearchSYCL<T, int32_t>(
                points, points_row_splits, queries, queries_row_splits, knn,
                neighbors_index, neighbors_row_splits, neighbors_distance,
                tile_bytes, max_tile_queries, tile_points_alignment,
                /*force_addmm_path=*/true);
    };

    try {
        run_once();  // Warm-up; surfaces invalid tile shapes early.
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
        return;
    }

    for (auto _ : state) {
        run_once();
    }
}

void NNS_KnnAddMMTuneFloat(benchmark::State& state,
                           int64_t num_points,
                           int64_t num_queries,
                           int64_t dim,
                           int knn) {
    NNS_KnnAddMMTuneT<float>(state, num_points, num_queries, dim, knn);
}

// 1M points, 1000 queries (matches the NNS_KnnSearch_1M_k* baseline).
// Capture A: dim=16 (> kKnnDirectMaxDim), knn=8 -- forces AddMM via dim.
// Capture B: dim=3, knn=64 (> kSYCLKnnSmallKMax) -- forces AddMM via knn.
BENCHMARK_CAPTURE(NNS_KnnAddMMTuneFloat,
                  SYCL_KnnAddMMTune_1M_dim16_k8,
                  1000000,
                  1000,
                  16,
                  8)
        ->ArgNames({"tile_MiB", "max_tile_q", "tile_pt_align"})
        ->ArgsProduct({{1, 2, 4, 8, 16, 32, 64},
                       {32, 64, 128, 256},
                       {64, 128, 256}})
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(NNS_KnnAddMMTuneFloat,
                  SYCL_KnnAddMMTune_1M_dim3_k64,
                  1000000,
                  1000,
                  3,
                  64)
        ->ArgNames({"tile_MiB", "max_tile_q", "tile_pt_align"})
        ->ArgsProduct({{1, 2, 4, 8, 16, 32, 64},
                       {32, 64, 128, 256},
                       {64, 128, 256}})
        ->Unit(benchmark::kMillisecond);

// Follow-up: max_tile_queries dominated Part 1 (monotonically better up to
// the largest tested value, 256), while tile_bytes/tile_pt_align barely
// mattered (<2%). This narrow sweep extends max_tile_queries beyond 256 (at
// a fixed, near-optimal tile_bytes=8MiB/align=128) to find where it
// plateaus, since num_queries=1000 means values >=1000 disable query tiling
// entirely.
BENCHMARK_CAPTURE(NNS_KnnAddMMTuneFloat,
                  SYCL_KnnAddMMTune_1M_dim16_k8_bigq,
                  1000000,
                  1000,
                  16,
                  8)
        ->ArgNames({"tile_MiB", "max_tile_q", "tile_pt_align"})
        ->ArgsProduct({{8}, {256, 512, 1024, 2048}, {128}})
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(NNS_KnnAddMMTuneFloat,
                  SYCL_KnnAddMMTune_1M_dim3_k64_bigq,
                  1000000,
                  1000,
                  3,
                  64)
        ->ArgNames({"tile_MiB", "max_tile_q", "tile_pt_align"})
        ->ArgsProduct({{8}, {256, 512, 1024, 2048}, {128}})
        ->Unit(benchmark::kMillisecond);

// Follow-up 2: validate at a much larger, more realistic query count
// (100K queries against 1M points -- closer to a real self-KNN workload)
// that increasing max_tile_queries keeps helping (or at least doesn't
// regress/blow up memory) rather than being an artifact of the small
// 1000-query capture above.
BENCHMARK_CAPTURE(NNS_KnnAddMMTuneFloat,
                  SYCL_KnnAddMMTune_1M_100Kq_dim16_k8_bigq,
                  1000000,
                  100000,
                  16,
                  8)
        ->ArgNames({"tile_MiB", "max_tile_q", "tile_pt_align"})
        ->ArgsProduct({{8}, {128, 256, 512, 1024, 2048, 4096, 8192}, {128}})
        ->Unit(benchmark::kMillisecond);

// ── Part 2: direct-vs-AddMM boundary comparison ────────────────────────────
// Args (via ->ArgsProduct):
//   0: dim
//   1: knn
//   2: force_addmm (0 = production dispatch via UseKnnDirect picks
//      whichever path is valid/faster by design; 1 = force AddMM even when
//      (dim, knn) would qualify for direct, to A/B against the same point).
// tile_bytes/max_tile_queries/tile_points_alignment are fixed to the winning
// combo found in Part 1's sweep (see plan.md notes / PR description for the
// exact chosen values).
template <typename T>
void NNS_KnnPathCompareT(benchmark::State& state,
                         int64_t num_points,
                         int64_t num_queries,
                         int64_t tile_bytes,
                         int64_t max_tile_queries,
                         int64_t tile_points_alignment) {
    const int64_t dim = state.range(0);
    const int knn = static_cast<int>(state.range(1));
    const bool force_addmm = state.range(2) != 0;

    const core::Device device("SYCL:0");
    const core::Dtype dtype = core::Dtype::FromType<T>();
    const core::Tensor points =
            benchmarks::Rand({num_points, dim}, 42, {0.0, 1.0}, dtype, device);
    const core::Tensor queries =
            benchmarks::Rand({num_queries, dim}, 43, {0.0, 1.0}, dtype, device);
    const core::Tensor points_row_splits =
            core::Tensor::Init<int64_t>({0, num_points});
    const core::Tensor queries_row_splits =
            core::Tensor::Init<int64_t>({0, num_queries});

    auto run_once = [&]() {
        core::Tensor neighbors_index, neighbors_distance;
        core::Tensor neighbors_row_splits =
                core::Tensor::Empty({num_queries + 1}, core::Int64);
        core::nns::KnnSearchSYCL<T, int32_t>(
                points, points_row_splits, queries, queries_row_splits, knn,
                neighbors_index, neighbors_row_splits, neighbors_distance,
                tile_bytes, max_tile_queries, tile_points_alignment,
                force_addmm);
    };

    try {
        run_once();
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
        return;
    }

    for (auto _ : state) {
        run_once();
    }
}

void NNS_KnnPathCompareFloat(benchmark::State& state,
                             int64_t num_points,
                             int64_t num_queries,
                             int64_t tile_bytes,
                             int64_t max_tile_queries,
                             int64_t tile_points_alignment) {
    NNS_KnnPathCompareT<float>(state, num_points, num_queries, tile_bytes,
                               max_tile_queries, tile_points_alignment);
}

// Tile-shape args from the Part 1 sweep + its bigq/100Kq follow-ups:
// max_tile_queries dominates (up to 8x, monotonic 32->8192), while
// tile_bytes and tile_pt_align each move the result by <2%. 2048 is chosen
// over the true plateau (~4096-8192) because at tile_bytes=8MiB/float32,
// ChooseTileSize's tile_points floors at 256 once tile_queries exceeds
// ~8192 (tile_points ~= tile_bytes/(tile_queries*element_size)), after
// which larger max_tile_queries stops reducing GEMM-tile count but keeps
// growing the -2qp tile's memory footprint past the tile_bytes budget;
// 2048 keeps tile_points a comfortable 4x above that floor while already
// capturing ~98% of the achievable speedup.
constexpr int64_t kAddMMTunedTileBytes = 8LL * 1024 * 1024;
constexpr int64_t kAddMMTunedMaxTileQueries = 2048;
constexpr int64_t kAddMMTunedTilePointsAlignment = 128;

BENCHMARK_CAPTURE(NNS_KnnPathCompareFloat,
                  SYCL_KnnPathCompare_1M,
                  1000000,
                  1000,
                  kAddMMTunedTileBytes,
                  kAddMMTunedMaxTileQueries,
                  kAddMMTunedTilePointsAlignment)
        ->ArgNames({"dim", "k", "force_addmm"})
        ->ArgsProduct({{1, 2, 3, 4, 6, 8}, {1, 4, 8, 16, 32}, {0, 1}})
        ->Unit(benchmark::kMillisecond);

// ── Part 3: direct-vs-AddMM across num_points scale ────────────────────────
// SYCL_KnnPathCompare_1M (Part 2) only tested num_points=1M. This sweeps
// num_points itself (at a fixed representative dim/knn) to check whether
// Direct's win margin holds, shrinks, or reverses at very small/large point
// counts -- i.e. whether the boundary conclusion is scale-invariant.
// Args (via ->ArgsProduct):
//   0: num_points
//   1: force_addmm (0 = production dispatch, picks Direct; 1 = force AddMM)
template <typename T>
void NNS_KnnPathScaleT(benchmark::State& state,
                       int64_t dim,
                       int knn,
                       int64_t tile_bytes,
                       int64_t max_tile_queries,
                       int64_t tile_points_alignment) {
    const int64_t num_points = state.range(0);
    const bool force_addmm = state.range(1) != 0;
    const int64_t num_queries = std::min<int64_t>(num_points, 1000);

    const core::Device device("SYCL:0");
    const core::Dtype dtype = core::Dtype::FromType<T>();
    const core::Tensor points =
            benchmarks::Rand({num_points, dim}, 42, {0.0, 1.0}, dtype, device);
    const core::Tensor queries =
            benchmarks::Rand({num_queries, dim}, 43, {0.0, 1.0}, dtype, device);
    const core::Tensor points_row_splits =
            core::Tensor::Init<int64_t>({0, num_points});
    const core::Tensor queries_row_splits =
            core::Tensor::Init<int64_t>({0, num_queries});

    auto run_once = [&]() {
        core::Tensor neighbors_index, neighbors_distance;
        core::Tensor neighbors_row_splits =
                core::Tensor::Empty({num_queries + 1}, core::Int64);
        core::nns::KnnSearchSYCL<T, int32_t>(
                points, points_row_splits, queries, queries_row_splits, knn,
                neighbors_index, neighbors_row_splits, neighbors_distance,
                tile_bytes, max_tile_queries, tile_points_alignment,
                force_addmm);
    };

    try {
        run_once();
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
        return;
    }

    for (auto _ : state) {
        run_once();
    }
}

void NNS_KnnPathScaleFloat(benchmark::State& state,
                           int64_t dim,
                           int knn,
                           int64_t tile_bytes,
                           int64_t max_tile_queries,
                           int64_t tile_points_alignment) {
    NNS_KnnPathScaleT<float>(state, dim, knn, tile_bytes, max_tile_queries,
                             tile_points_alignment);
}

// dim=3/k=8: typical point-cloud KNN case.
BENCHMARK_CAPTURE(NNS_KnnPathScaleFloat,
                  SYCL_KnnPathScale_dim3_k8,
                  3,
                  8,
                  kAddMMTunedTileBytes,
                  kAddMMTunedMaxTileQueries,
                  kAddMMTunedTilePointsAlignment)
        ->ArgNames({"num_points", "force_addmm"})
        ->ArgsProduct({{100, 1000, 10000, 100000, 1000000, 5000000}, {0, 1}})
        ->Unit(benchmark::kMillisecond);

// dim=8/k=32: edge of Direct's valid domain -- smallest observed win margin
// in Part 2 (4.5x-5x at 1M points), so the most likely place for the
// boundary conclusion to flip at a different scale.
BENCHMARK_CAPTURE(NNS_KnnPathScaleFloat,
                  SYCL_KnnPathScale_dim8_k32,
                  8,
                  32,
                  kAddMMTunedTileBytes,
                  kAddMMTunedMaxTileQueries,
                  kAddMMTunedTilePointsAlignment)
        ->ArgNames({"num_points", "force_addmm"})
        ->ArgsProduct({{100, 1000, 10000, 100000, 1000000, 5000000}, {0, 1}})
        ->Unit(benchmark::kMillisecond);

}  // namespace

}  // namespace nns
}  // namespace benchmarks
}  // namespace open3d

#endif  // BUILD_SYCL_MODULE
