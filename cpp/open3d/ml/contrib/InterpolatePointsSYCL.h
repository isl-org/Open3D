// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// SYCL implementation of three_nn / three_interpolate(_grad) — ports
// InterpolatePoints.cu.
//
// ThreeNNSYCL: one work-group per (batch, query); work-items grid-stride over
// candidates, merge per-lane top-3 in SLM, then work-item 0 merges to the
// final top-3. Equal-distance tie-breaking differs from a single-work-item
// serial scan (CUDA reference); python/test/ml_ops tests check distances and
// the three neighbor indices, not CUDA's exact tie order. A serial-scan port
// would match CUDA tie order but is not required today.
//
// three_interpolate / _grad: order-independent gather and atomic scatter
// (InvertNeighborsList pattern), launched via core::ParallelFor.

#pragma once

#include <sycl/sycl.hpp>

#include "open3d/core/ParallelFor.h"

namespace open3d {
namespace ml {
namespace contrib {

namespace {
// Work-group size for ThreeNNSYCL: one work-group per query, work-items
// grid-stride over the m candidate points. Best-guess default (matches the
// work-group-per-output-point size used elsewhere in this codebase, e.g.
// BallQuerySYCL.h/the conv FillColumn kernels); not yet tuned on target HW.
constexpr size_t kThreeNNWGSize = 32;
}  // namespace

/// Finds, for each of the n "unknown" points, the 3 nearest of the m "known"
/// points (per batch element). Ports three_nn_kernel.
///
/// \param queue      SYCL queue to run the kernel on.
/// \param unknown    Query point positions, shape [b, n, 3].
/// \param known      Reference point positions, shape [b, m, 3].
/// \param dist2      Output squared distances to the 3 nearest, shape
///        [b, n, 3].
/// \param idx        Output indices of the 3 nearest, shape [b, n, 3].
inline void ThreeNNSYCL(sycl::queue& queue,
                        int b,
                        int n,
                        int m,
                        const float* const unknown,
                        const float* const known,
                        float* const dist2,
                        int* const idx) {
    if (b <= 0 || n <= 0) return;
    const size_t wg = kThreeNNWGSize;

    queue.submit([&](sycl::handler& cgh) {
        // Per-work-item local top-3 (value, index), staged here so work-item
        // 0 can merge all work-items' candidates after the barrier below.
        sycl::local_accessor<double, 1> local_best(3 * wg, cgh);
        sycl::local_accessor<int, 1> local_besti(3 * wg, cgh);

        cgh.parallel_for(
                sycl::nd_range<1>(
                        sycl::range<1>(static_cast<size_t>(b) * n * wg),
                        sycl::range<1>(wg)),
                // Distinct buffers — safe for [[intel::kernel_args_restrict]].
                [=](sycl::nd_item<1> item) [[intel::kernel_args_restrict]] {
                    const size_t group_id = item.get_group(0);
                    const int bs_idx = static_cast<int>(group_id / n);
                    const int pt_idx = static_cast<int>(group_id % n);
                    const size_t lid = item.get_local_id(0);
                    auto group = item.get_group();

                    const float* const u =
                            unknown + bs_idx * n * 3 + pt_idx * 3;
                    const float* const kn = known + bs_idx * m * 3;

                    const float ux = u[0];
                    const float uy = u[1];
                    const float uz = u[2];

                    double best1 = 1e40, best2 = 1e40, best3 = 1e40;
                    int besti1 = 0, besti2 = 0, besti3 = 0;
                    for (int k = static_cast<int>(lid); k < m;
                         k += static_cast<int>(wg)) {
                        const float x = kn[k * 3 + 0];
                        const float y = kn[k * 3 + 1];
                        const float z = kn[k * 3 + 2];
                        const double d = double((ux - x) * (ux - x) +
                                                (uy - y) * (uy - y) +
                                                (uz - z) * (uz - z));
                        if (d < best1) {
                            best3 = best2;
                            besti3 = besti2;
                            best2 = best1;
                            besti2 = besti1;
                            best1 = d;
                            besti1 = k;
                        } else if (d < best2) {
                            best3 = best2;
                            besti3 = besti2;
                            best2 = d;
                            besti2 = k;
                        } else if (d < best3) {
                            best3 = d;
                            besti3 = k;
                        }
                    }

                    local_best[3 * lid + 0] = best1;
                    local_best[3 * lid + 1] = best2;
                    local_best[3 * lid + 2] = best3;
                    local_besti[3 * lid + 0] = besti1;
                    local_besti[3 * lid + 1] = besti2;
                    local_besti[3 * lid + 2] = besti3;
                    sycl::group_barrier(group);

                    if (lid != 0) return;

                    // Merge up to 3*wg per-work-item candidates (some may be
                    // the 1e40 sentinel if that work-item's slice had fewer
                    // than 3 candidates, e.g. when m < wg) into the final
                    // top-3, using the same insertion-style update as the
                    // per-work-item scan above.
                    float* const d2_out = dist2 + bs_idx * n * 3 + pt_idx * 3;
                    int* const idx_out = idx + bs_idx * n * 3 + pt_idx * 3;

                    double mbest1 = 1e40, mbest2 = 1e40, mbest3 = 1e40;
                    int mbesti1 = 0, mbesti2 = 0, mbesti3 = 0;
                    for (size_t c = 0; c < 3 * wg; ++c) {
                        const double d = local_best[c];
                        const int di = local_besti[c];
                        if (d < mbest1) {
                            mbest3 = mbest2;
                            mbesti3 = mbesti2;
                            mbest2 = mbest1;
                            mbesti2 = mbesti1;
                            mbest1 = d;
                            mbesti1 = di;
                        } else if (d < mbest2) {
                            mbest3 = mbest2;
                            mbesti3 = mbesti2;
                            mbest2 = d;
                            mbesti2 = di;
                        } else if (d < mbest3) {
                            mbest3 = d;
                            mbesti3 = di;
                        }
                    }
                    d2_out[0] = static_cast<float>(mbest1);
                    d2_out[1] = static_cast<float>(mbest2);
                    d2_out[2] = static_cast<float>(mbest3);
                    idx_out[0] = mbesti1;
                    idx_out[1] = mbesti2;
                    idx_out[2] = mbesti3;
                });
    });
    queue.wait_and_throw();
}

/// Weighted gather of 3 nearest neighbor features. Ports
/// three_interpolate_kernel.
///
/// \param points     Source features, shape [b, c, m].
/// \param idx        Indices of the 3 nearest, shape [b, n, 3].
/// \param weight     Interpolation weights, shape [b, n, 3].
/// \param out        Output features, shape [b, c, n].
inline void ThreeInterpolateSYCL(sycl::queue& queue,
                                 int b,
                                 int c,
                                 int m,
                                 int n,
                                 const float* const points,
                                 const int* const idx,
                                 const float* const weight,
                                 float* const out) {
    if (b <= 0 || c <= 0 || n <= 0) return;

    core::ParallelFor(
            queue, static_cast<int64_t>(b) * c * n, [=](int64_t idx64) {
                const int bs_idx = static_cast<int>(idx64 / (c * n));
                const int rem = static_cast<int>(idx64 % (c * n));
                const int c_idx = rem / n;
                const int pt_idx = rem % n;

                const float* const w = weight + bs_idx * n * 3 + pt_idx * 3;
                const float* const pts = points + bs_idx * c * m + c_idx * m;
                const int* const id = idx + bs_idx * n * 3 + pt_idx * 3;
                float* const o = out + bs_idx * c * n + c_idx * n;

                o[pt_idx] = w[0] * pts[id[0]] + w[1] * pts[id[1]] +
                            w[2] * pts[id[2]];
            });
}

/// Gradient of ThreeInterpolateSYCL w.r.t. points. Scatters (atomically
/// accumulates) grad_out * weight into grad_points at the 3 neighbor
/// indices. grad_points must be zero-initialized by the caller before this
/// call. Ports three_interpolate_grad_kernel.
inline void ThreeInterpolateGradSYCL(sycl::queue& queue,
                                     int b,
                                     int c,
                                     int n,
                                     int m,
                                     const float* const grad_out,
                                     const int* const idx,
                                     const float* const weight,
                                     float* const grad_points) {
    if (b <= 0 || c <= 0 || n <= 0) return;

    core::ParallelFor(
            queue, static_cast<int64_t>(b) * c * n, [=](int64_t idx64) {
                const int bs_idx = static_cast<int>(idx64 / (c * n));
                const int rem = static_cast<int>(idx64 % (c * n));
                const int c_idx = rem / n;
                const int pt_idx = rem % n;

                const float g = grad_out[bs_idx * c * n + c_idx * n + pt_idx];
                const float* const w = weight + bs_idx * n * 3 + pt_idx * 3;
                const int* const id = idx + bs_idx * n * 3 + pt_idx * 3;
                float* const gp = grad_points + bs_idx * c * m + c_idx * m;

                for (int l = 0; l < 3; ++l) {
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                            ref(gp[id[l]]);
                    ref.fetch_add(g * w[l]);
                }
            });
}

}  // namespace contrib
}  // namespace ml
}  // namespace open3d
