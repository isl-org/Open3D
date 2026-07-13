// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// SYCL implementation + XPU dispatch wrapper for FurthestPointSampling —
// ports PointSampling.cuh's furthest_point_sampling_kernel. The CUDA kernel
// does a value+index argmax tree reduction in shared memory (not a plain
// sum); replicated here with sycl::local_accessor + nd_item::barrier, one
// work-group per batch element.

#include <c10/xpu/XPUStream.h>

#include <algorithm>

#include "open3d/ml/pytorch/pointnet/SamplingKernel.h"

namespace {

/// Iteratively picks m points from n candidates per batch that are furthest
/// from the already-selected set (greedy farthest point sampling).
///
/// \param queue      SYCL queue.
/// \param b          Batch size.
/// \param n          Number of candidate points per batch.
/// \param m          Number of points to sample per batch.
/// \param dataset    Point positions, shape [b, n, 3].
/// \param temp       Scratch distance buffer, shape [b, n]. Caller must
///        initialize to a large value (matches CUDA convention: 1e10).
/// \param idxs       Output sampled indices, shape [b, m].
/// \param work_group_size    Work-group size (tunable hyperparameter);
///        capped internally to the device's max and to a power of two.
void FurthestPointSamplingSYCL(sycl::queue& queue,
                               int b,
                               int n,
                               int m,
                               const float* const dataset,
                               float* const temp,
                               int* const idxs,
                               size_t work_group_size = 256) {
    if (b <= 0 || m <= 0) return;

    const size_t max_wg =
            queue.get_device()
                    .get_info<sycl::info::device::max_work_group_size>();
    size_t wg = std::min(work_group_size, max_wg);
    // Round down to a power of two so the tree reduction is exact.
    size_t pow2 = 1;
    while (pow2 * 2 <= wg) pow2 *= 2;
    wg = std::max<size_t>(pow2, 1);

    const sycl::range<1> global(static_cast<size_t>(b) * wg);
    const sycl::range<1> local(wg);

    queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> dists(wg, cgh);
        sycl::local_accessor<int, 1> dists_i(wg, cgh);

        cgh.parallel_for(
                sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> item) {
                    const int batch_index = static_cast<int>(item.get_group(0));
                    const int tid = static_cast<int>(item.get_local_id(0));
                    const int stride =
                            static_cast<int>(item.get_local_range(0));

                    const float* const data = dataset + batch_index * n * 3;
                    float* const tmp = temp + batch_index * n;
                    int* const out_idx = idxs + batch_index * m;

                    int old = 0;
                    if (tid == 0) out_idx[0] = old;
                    item.barrier(sycl::access::fence_space::local_space);

                    for (int j = 1; j < m; ++j) {
                        int besti = 0;
                        float best = -1;
                        const float x1 = data[old * 3 + 0];
                        const float y1 = data[old * 3 + 1];
                        const float z1 = data[old * 3 + 2];
                        for (int k = tid; k < n; k += stride) {
                            const float x2 = data[k * 3 + 0];
                            const float y2 = data[k * 3 + 1];
                            const float z2 = data[k * 3 + 2];
                            const float d = (x2 - x1) * (x2 - x1) +
                                            (y2 - y1) * (y2 - y1) +
                                            (z2 - z1) * (z2 - z1);
                            const float d2 = sycl::min(d, tmp[k]);
                            tmp[k] = d2;
                            besti = d2 > best ? k : besti;
                            best = d2 > best ? d2 : best;
                        }
                        dists[tid] = best;
                        dists_i[tid] = besti;
                        item.barrier(sycl::access::fence_space::local_space);

                        // Argmax tree reduction over [dists, dists_i] pairs.
                        for (size_t s = wg / 2; s > 0; s >>= 1) {
                            if (static_cast<size_t>(tid) < s) {
                                const float v1 = dists[tid];
                                const float v2 = dists[tid + s];
                                const int i1 = dists_i[tid];
                                const int i2 = dists_i[tid + s];
                                dists[tid] = sycl::max(v1, v2);
                                dists_i[tid] = v2 > v1 ? i2 : i1;
                            }
                            item.barrier(
                                    sycl::access::fence_space::local_space);
                        }

                        old = dists_i[0];
                        if (tid == 0) out_idx[j] = old;
                        // Extra barrier vs. the CUDA original: SYCL
                        // work-item scheduling does not guarantee
                        // CUDA-warp-like lock-step progress, so make sure
                        // every work-item has consumed dists_i[0] before the
                        // next iteration overwrites it.
                        item.barrier(sycl::access::fence_space::local_space);
                    }
                });
    });
    queue.wait_and_throw();
}

}  // namespace

void furthest_point_sampling_launcher_sycl(
        int b, int n, int m, const float* dataset, float* temp, int* idxs) {
    sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();
    FurthestPointSamplingSYCL(queue, b, n, m, dataset, temp, idxs);
}
