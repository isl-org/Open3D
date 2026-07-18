// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// SYCL implementation of BallQuery — ports BallQuery.cu (ball_query_kernel).
// One work-item per (batch, query point); serial scan over candidate points
// up to nsample. No reduction needed.

#pragma once

#include <sycl/sycl.hpp>

namespace open3d {
namespace ml {
namespace contrib {

/// Finds up to nsample points within radius of each query point.
///
/// \param queue      SYCL queue (PyTorch XPU stream queue).
/// \param b          Batch size.
/// \param n          Number of candidate points per batch (xyz).
/// \param m          Number of query points per batch (new_xyz).
/// \param radius     Search radius.
/// \param nsample    Max number of neighbors to record per query point.
/// \param new_xyz    Query point positions, shape [b, m, 3].
/// \param xyz        Candidate point positions, shape [b, n, 3].
/// \param idx        Output neighbor indices, shape [b, m, nsample].
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

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
                sycl::range<2>(static_cast<size_t>(b), static_cast<size_t>(m)),
                [=](sycl::item<2> item) {
                    const int bs_idx = static_cast<int>(item.get_id(0));
                    const int pt_idx = static_cast<int>(item.get_id(1));

                    const float* const nxyz =
                            new_xyz + bs_idx * m * 3 + pt_idx * 3;
                    const float* const xyz_batch = xyz + bs_idx * n * 3;
                    int* const idx_out =
                            idx + bs_idx * m * nsample + pt_idx * nsample;

                    const float new_x = nxyz[0];
                    const float new_y = nxyz[1];
                    const float new_z = nxyz[2];

                    int cnt = 0;
                    for (int k = 0; k < n; ++k) {
                        const float x = xyz_batch[k * 3 + 0];
                        const float y = xyz_batch[k * 3 + 1];
                        const float z = xyz_batch[k * 3 + 2];
                        const float d2 = (new_x - x) * (new_x - x) +
                                         (new_y - y) * (new_y - y) +
                                         (new_z - z) * (new_z - z);
                        if (d2 < radius2) {
                            if (cnt == 0) {
                                for (int l = 0; l < nsample; ++l) {
                                    idx_out[l] = k;
                                }
                            }
                            idx_out[cnt] = k;
                            ++cnt;
                            if (cnt >= nsample) break;
                        }
                    }
                });
    });
    queue.wait_and_throw();
}

}  // namespace contrib
}  // namespace ml
}  // namespace open3d
