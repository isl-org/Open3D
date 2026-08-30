// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// SYCL implementation of BallQuery — ports BallQuery.cu (ball_query_kernel).
// One work-group per (batch, query); work-items grid-stride over candidates
// and use exclusive_scan_over_group to claim output slots. When more than
// nsample points are in range, the in-range neighbor set is correct but slot
// order may differ from CUDA; python/test/ml_ops/test_query_pts.py compares
// neighbor sets, not order. A serial-scan kernel would preserve CUDA index
// order but is not required today. Padding when total matches < nsample still
// duplicates idx_out[0] into remaining slots (CUDA contract).
#pragma once

#include <sycl/sycl.hpp>

namespace open3d {
namespace ml {
namespace contrib {

namespace {
// Work-group size: one work-group per query point, work-items grid-stride
// over the n candidate points. Best-guess default (matches the
// work-group-per-output-point size used by the conv FillColumn kernels,
// which is itself a mirror of the CUDA warp-per-point design); not yet tuned
// on target HW for this op's access pattern.
constexpr size_t kBallQueryWGSize = 32;
}  // namespace

/// Finds up to nsample points within radius of each query point.
///
/// \param queue      SYCL queue to run the kernel on.
/// \param b          Batch size.
/// \param n          Number of candidate points per batch (xyz).
/// \param m          Number of query points per batch (new_xyz).
/// \param radius     Search radius.
/// \param nsample    Max number of neighbors to record per query point.
/// \param new_xyz    Query point positions, shape [b, m, 3].
/// \param xyz        Candidate point positions, shape [b, n, 3].
/// \param idx        Output neighbor indices, shape [b, m, nsample]. Must be
/// zero-initialized by the caller (matches the CUDA/original-SYCL contract
/// of leaving index 0 as the implicit match when no candidate is in range).
inline void BallQuerySYCL(sycl::queue& queue,
                          int b,
                          int n,
                          int m,
                          float radius,
                          int nsample,
                          const float* const new_xyz,
                          const float* const xyz,
                          int* const idx) {
    if (b <= 0 || m <= 0) return;
    const float radius2 = radius * radius;
    const size_t wg = kBallQueryWGSize;

    queue.submit([&](sycl::handler& cgh) {
        // Per-work-item running match count, used to derive each work-item's
        // base output slot via an exclusive scan before writing.
        cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(size_t(b) * m * wg),
                                  sycl::range<1>(wg)),
                // Distinct buffers — safe for [[intel::kernel_args_restrict]].
                [=](sycl::nd_item<1> item) [[intel::kernel_args_restrict]] {
                    const size_t group_id = item.get_group(0);
                    const int bs_idx = static_cast<int>(group_id / m);
                    const int pt_idx = static_cast<int>(group_id % m);
                    const size_t lid = item.get_local_id(0);
                    auto group = item.get_group();

                    const float* const nxyz =
                            new_xyz + bs_idx * m * 3 + pt_idx * 3;
                    const float* const xyz_batch = xyz + bs_idx * n * 3;
                    int* const idx_out =
                            idx + bs_idx * m * nsample + pt_idx * nsample;

                    const float new_x = nxyz[0];
                    const float new_y = nxyz[1];
                    const float new_z = nxyz[2];

                    // Local match count for this work-item's strided slice.
                    int local_count = 0;
                    for (int k = static_cast<int>(lid); k < n;
                         k += static_cast<int>(wg)) {
                        const float x = xyz_batch[k * 3 + 0];
                        const float y = xyz_batch[k * 3 + 1];
                        const float z = xyz_batch[k * 3 + 2];
                        const float d2 = (new_x - x) * (new_x - x) +
                                         (new_y - y) * (new_y - y) +
                                         (new_z - z) * (new_z - z);
                        if (d2 < radius2) ++local_count;
                    }

                    // Exclusive scan of local_count across the work-group
                    // gives each work-item's base output slot; the group's
                    // total (via a matching inclusive/exclusive-plus-local
                    // sum) is used below to detect the zero-match case.
                    const int base_slot = sycl::exclusive_scan_over_group(
                            group, local_count, sycl::plus<int>());
                    const int total_count = sycl::reduce_over_group(
                            group, local_count, sycl::plus<int>());

                    if (total_count == 0) {
                        // No candidate in range for this query: idx_out is
                        // already zero-initialized by the caller, matching
                        // the original serial kernel leaving it untouched.
                        return;
                    }

                    // Second pass: re-scan this work-item's slice, writing
                    // matches at [base_slot, base_slot+1, ...) capped at
                    // nsample, mirroring the serial kernel's cnt/idx_out[cnt]
                    // bookkeeping (but slots are now assigned by the scan
                    // instead of a running scalar counter).
                    int slot = base_slot;
                    for (int k = static_cast<int>(lid); k < n && slot < nsample;
                         k += static_cast<int>(wg)) {
                        const float x = xyz_batch[k * 3 + 0];
                        const float y = xyz_batch[k * 3 + 1];
                        const float z = xyz_batch[k * 3 + 2];
                        const float d2 = (new_x - x) * (new_x - x) +
                                         (new_y - y) * (new_y - y) +
                                         (new_z - z) * (new_z - z);
                        if (d2 < radius2) {
                            idx_out[slot] = k;
                            ++slot;
                        }
                    }

                    // Pad slots [total_count, nsample) with idx_out[0]; barrier
                    // so work-item 0 reads a value another lane may have
                    // written.
                    sycl::group_barrier(group);
                    if (lid == 0 && total_count < nsample) {
                        const int first_value = idx_out[0];
                        for (int l = total_count; l < nsample; ++l) {
                            idx_out[l] = first_value;
                        }
                    }
                });
    });
    queue.wait_and_throw();
}

}  // namespace contrib
}  // namespace ml
}  // namespace open3d
