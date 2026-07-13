// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// SYCL implementation of InvertNeighborsList — ports InvertNeighborsList.cuh.
// Replaces: atomicAdd → sycl::atomic_ref::fetch_add,
//           cub::DeviceScan::InclusiveSum → std::inclusive_scan (oneDPL),
//           cudaMemsetAsync → queue.fill.

#pragma once

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <sycl/sycl.hpp>

namespace open3d {
namespace ml {
namespace impl {

/// Count the number of occurrences of each index value using atomic add.
/// count[] must be zero-initialized before the call.
/// Named CountIndexOccurrencesSYCL to distinguish from the unrelated
/// CountNeighborsSYCL in open3d::core::nns (which counts FRS neighbors).
template <class T>
void CountIndexOccurrencesSYCL(sycl::queue& queue,
                        uint32_t* count,
                        size_t count_size,
                        const T* indices,
                        size_t indices_size) {
    queue.fill(count, uint32_t(0), count_size).wait();
    if (indices_size == 0) return;

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(indices_size), [=](sycl::item<1> item) {
            T idx = indices[item.get_id(0)];
            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                    ref(count[idx]);
            ref.fetch_add(1u);
        });
    });
    queue.wait_and_throw();
}

/// Fill output index and attribute arrays for the inverted list.
/// Uses atomic_ref to claim positions within each output row.
template <class TIndex, class TAttr>
void FillNeighborsIndexAndAttributesSYCL(
        sycl::queue& queue,
        uint32_t* count,          // scratch: reused as write-position counter
        size_t count_size,
        TIndex* out_neighbors_index,
        TAttr* out_neighbors_attributes,
        const TIndex* inp_neighbors_index,
        const TAttr* inp_neighbors_attributes,
        int num_attributes_per_neighbor,
        size_t index_size,
        const int64_t* inp_neighbors_row_splits,
        size_t inp_num_queries,
        const int64_t* out_neighbors_row_splits,
        size_t out_num_queries) {
    queue.fill(count, uint32_t(0), count_size).wait();
    if (inp_num_queries == 0) return;

    const bool fill_attributes = (inp_neighbors_attributes != nullptr) &&
                                 (num_attributes_per_neighbor > 0);
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
                sycl::range<1>(inp_num_queries), [=](sycl::item<1> item) {
                    const size_t i = item.get_id(0);
                    const size_t begin_idx =
                            static_cast<size_t>(inp_neighbors_row_splits[i]);
                    const size_t end_idx = static_cast<size_t>(
                            inp_neighbors_row_splits[i + 1]);

                    for (size_t j = begin_idx; j < end_idx; ++j) {
                        TIndex neighbor_idx = inp_neighbors_index[j];
                        size_t list_offset = static_cast<size_t>(
                                out_neighbors_row_splits[neighbor_idx]);
                        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                                ref(count[neighbor_idx]);
                        const size_t item_offset = ref.fetch_add(1u);

                        out_neighbors_index[list_offset + item_offset] =
                                static_cast<TIndex>(i);

                        if (fill_attributes) {
                            TAttr* dst = out_neighbors_attributes +
                                         num_attributes_per_neighbor *
                                                 (list_offset + item_offset);
                            const TAttr* src =
                                    inp_neighbors_attributes +
                                    num_attributes_per_neighbor * j;
                            for (int a = 0; a < num_attributes_per_neighbor;
                                 ++a) {
                                dst[a] = src[a];
                            }
                        }
                    }
                });
    });
    queue.wait_and_throw();
}

/// Inverts a neighbors list.
///
/// All pointer arguments must point to SYCL device-accessible memory (USM or
/// PyTorch XPU tensors).
///
/// count_buf must be device memory of at least out_num_queries uint32_t
/// elements (used as a scratch buffer, zero-initialized internally).
template <class TIndex, class TAttr>
void InvertNeighborsListSYCL(sycl::queue& queue,
                              uint32_t* count_buf,
                              const TIndex* inp_neighbors_index,
                              const TAttr* inp_neighbors_attributes,
                              int num_attributes_per_neighbor,
                              const int64_t* inp_neighbors_row_splits,
                              size_t inp_num_queries,
                              TIndex* out_neighbors_index,
                              TAttr* out_neighbors_attributes,
                              size_t index_size,
                              int64_t* out_neighbors_row_splits,
                              size_t out_num_queries) {
    // Step 1: Count occurrences of each neighbor index → raw counts
    CountIndexOccurrencesSYCL(queue, count_buf, out_num_queries,
                               inp_neighbors_index, index_size);

    // Step 2: Compute inclusive prefix sum → out_neighbors_row_splits[1..]
    //         (out_neighbors_row_splits[0] = 0 is already zeroed by caller)
    auto dpl_policy = oneapi::dpl::execution::make_device_policy(queue);
    std::inclusive_scan(dpl_policy, count_buf, count_buf + out_num_queries,
                        out_neighbors_row_splits + 1);
    queue.wait_and_throw();

    // Step 3: Fill output indices and attributes using atomic position tracking
    FillNeighborsIndexAndAttributesSYCL(
            queue, count_buf, out_num_queries, out_neighbors_index,
            out_neighbors_attributes, inp_neighbors_index,
            inp_neighbors_attributes, num_attributes_per_neighbor, index_size,
            inp_neighbors_row_splits, inp_num_queries, out_neighbors_row_splits,
            out_num_queries);
}

}  // namespace impl
}  // namespace ml
}  // namespace open3d
