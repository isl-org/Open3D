// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// SYCL wrapper for FixedRadiusSearch PyTorch op. Implements count-then-gather
// using CountNeighborsSYCL + WriteNeighborsSYCL from the core NNS layer,
// with the PyTorch XPU stream queue. Only L2 metric is supported (matching
// the SYCL NNS backend).

#include <c10/xpu/XPUStream.h>

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>
#include <sycl/sycl.hpp>

#include "open3d/core/nns/NeighborSearchCommon.h"
#include "open3d/core/nns/kernel/FixedRadiusSearchSYCLImpl.h"
#include "open3d/ml/pytorch/TorchHelper.h"
#include "open3d/ml/pytorch/misc/NeighborSearchAllocator.h"
#include "torch/script.h"

using namespace open3d::core::nns;

template <class T, class TIndex>
void FixedRadiusSearchSYCL(const torch::Tensor& points,
                           const torch::Tensor& queries,
                           double radius,
                           const torch::Tensor& points_row_splits,
                           const torch::Tensor& queries_row_splits,
                           const torch::Tensor& hash_table_splits,
                           const torch::Tensor& hash_table_index,
                           const torch::Tensor& hash_table_cell_splits,
                           const Metric metric,
                           const bool ignore_query_point,
                           const bool return_distances,
                           torch::Tensor& neighbors_index,
                           torch::Tensor& neighbors_row_splits,
                           torch::Tensor& neighbors_distance) {
    TORCH_CHECK(metric == Metric::L2,
                "FixedRadiusSearch SYCL only supports L2 metric");

    sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();
    auto dpl_policy = oneapi::dpl::execution::make_device_policy(queue);

    const T radius_t = static_cast<T>(radius);
    const T threshold = radius_t * radius_t;  // compare squared distances
    const T inv_voxel_size = T(1) / (T(2) * radius_t);

    const T* points_ptr = points.data_ptr<T>();
    const T* queries_ptr = queries.data_ptr<T>();
    const uint32_t* hash_index_ptr = reinterpret_cast<const uint32_t*>(
            hash_table_index.data_ptr<int32_t>());

    const int64_t num_queries = queries.size(0);
    const int num_batches = static_cast<int>(queries_row_splits.size(0)) - 1;

    // Allocate count buffer (uint32_t per query, cast from int32)
    torch::Tensor counts_tensor =
            torch::empty({num_queries},
                         torch::dtype(torch::kInt32).device(queries.device()));
    uint32_t* counts_ptr =
            reinterpret_cast<uint32_t*>(counts_tensor.data_ptr<int32_t>());

    const int64_t* q_row_splits_ptr = queries_row_splits.data_ptr<int64_t>();
    const uint32_t* ht_splits_ptr = reinterpret_cast<const uint32_t*>(
            hash_table_splits.data_ptr<int32_t>());
    const uint32_t* cell_splits_ptr = reinterpret_cast<const uint32_t*>(
            hash_table_cell_splits.data_ptr<int32_t>());

    // Pass 1: count neighbors per query
    for (int b = 0; b < num_batches; ++b) {
        const int64_t query_begin = q_row_splits_ptr[b];
        const int64_t query_end = q_row_splits_ptr[b + 1];
        const uint32_t first_cell_idx = ht_splits_ptr[b];
        const uint32_t hash_table_size = ht_splits_ptr[b + 1] - first_cell_idx;

        CountNeighborsSYCL<T>(queue, counts_ptr + query_begin, hash_index_ptr,
                              cell_splits_ptr + first_cell_idx, hash_table_size,
                              queries_ptr + 3 * query_begin,
                              query_end - query_begin, points_ptr,
                              inv_voxel_size, radius_t, threshold);
    }
    queue.wait_and_throw();

    // Build row_splits from counts via inclusive scan
    neighbors_row_splits =
            torch::zeros({num_queries + 1},
                         torch::dtype(torch::kInt64).device(queries.device()));
    int64_t* row_splits_ptr = neighbors_row_splits.data_ptr<int64_t>();
    std::inclusive_scan(dpl_policy, counts_ptr, counts_ptr + num_queries,
                        row_splits_ptr + 1, std::plus<int64_t>(), int64_t(0));
    queue.wait_and_throw();

    // Read total neighbor count from device
    int64_t total_neighbors = 0;
    if (num_queries > 0) {
        queue.memcpy(&total_neighbors, row_splits_ptr + num_queries,
                     sizeof(int64_t))
                .wait_and_throw();
    }

    auto device_type = queries.device().type();
    auto device_idx = queries.device().index();
    NeighborSearchAllocator<T, TIndex> output_allocator(device_type,
                                                        device_idx);

    TIndex* out_indices_ptr;
    T* out_dist_ptr;
    output_allocator.AllocIndices(&out_indices_ptr, total_neighbors);
    output_allocator.AllocDistances(&out_dist_ptr,
                                    return_distances ? total_neighbors : 0);

    // Pass 2: write neighbor indices (and optionally distances)
    for (int b = 0; b < num_batches; ++b) {
        const int64_t query_begin = q_row_splits_ptr[b];
        const int64_t query_end = q_row_splits_ptr[b + 1];
        const uint32_t first_cell_idx = ht_splits_ptr[b];
        const uint32_t hash_table_size = ht_splits_ptr[b + 1] - first_cell_idx;

        WriteNeighborsSYCL<T, TIndex>(
                queue, out_indices_ptr, out_dist_ptr,
                row_splits_ptr + query_begin, hash_index_ptr,
                cell_splits_ptr + first_cell_idx, hash_table_size,
                queries_ptr + 3 * query_begin, query_end - query_begin,
                points_ptr, inv_voxel_size, radius_t, threshold,
                return_distances);
    }
    queue.wait_and_throw();

    neighbors_index = output_allocator.NeighborsIndex();
    neighbors_distance = output_allocator.NeighborsDistance();
}

#define INSTANTIATE(T, TIndex)                                                \
    template void FixedRadiusSearchSYCL<T, TIndex>(                           \
            const torch::Tensor& points, const torch::Tensor& queries,        \
            double radius, const torch::Tensor& points_row_splits,            \
            const torch::Tensor& queries_row_splits,                          \
            const torch::Tensor& hash_table_splits,                           \
            const torch::Tensor& hash_table_index,                            \
            const torch::Tensor& hash_table_cell_splits, const Metric metric, \
            const bool ignore_query_point, const bool return_distances,       \
            torch::Tensor& neighbors_index,                                   \
            torch::Tensor& neighbors_row_splits,                              \
            torch::Tensor& neighbors_distance);

INSTANTIATE(float, int32_t)
INSTANTIATE(float, int64_t)
