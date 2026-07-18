// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// SYCL implementation of three_nn / three_interpolate(_grad) — ports
// InterpolatePoints.cu. three_nn is a per-point top-3 nearest neighbor scan
// (serial, one work-item per point); three_interpolate is a weighted gather;
// the gradient uses atomic scatter (proven pattern from InvertNeighborsList).

#pragma once

#include <sycl/sycl.hpp>

namespace open3d {
namespace ml {
namespace contrib {

/// Finds, for each of the n "unknown" points, the 3 nearest of the m "known"
/// points (per batch element). Ports three_nn_kernel.
///
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

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
                sycl::range<2>(static_cast<size_t>(b), static_cast<size_t>(n)),
                [=](sycl::item<2> item) {
                    const int bs_idx = static_cast<int>(item.get_id(0));
                    const int pt_idx = static_cast<int>(item.get_id(1));

                    const float* const u =
                            unknown + bs_idx * n * 3 + pt_idx * 3;
                    const float* const kn = known + bs_idx * m * 3;
                    float* const d2_out = dist2 + bs_idx * n * 3 + pt_idx * 3;
                    int* const idx_out = idx + bs_idx * n * 3 + pt_idx * 3;

                    const float ux = u[0];
                    const float uy = u[1];
                    const float uz = u[2];

                    double best1 = 1e40, best2 = 1e40, best3 = 1e40;
                    int besti1 = 0, besti2 = 0, besti3 = 0;
                    for (int k = 0; k < m; ++k) {
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
                    d2_out[0] = static_cast<float>(best1);
                    d2_out[1] = static_cast<float>(best2);
                    d2_out[2] = static_cast<float>(best3);
                    idx_out[0] = besti1;
                    idx_out[1] = besti2;
                    idx_out[2] = besti3;
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

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
                sycl::range<3>(static_cast<size_t>(b), static_cast<size_t>(c),
                               static_cast<size_t>(n)),
                [=](sycl::item<3> item) {
                    const int bs_idx = static_cast<int>(item.get_id(0));
                    const int c_idx = static_cast<int>(item.get_id(1));
                    const int pt_idx = static_cast<int>(item.get_id(2));

                    const float* const w = weight + bs_idx * n * 3 + pt_idx * 3;
                    const float* const pts =
                            points + bs_idx * c * m + c_idx * m;
                    const int* const id = idx + bs_idx * n * 3 + pt_idx * 3;
                    float* const o = out + bs_idx * c * n + c_idx * n;

                    o[pt_idx] = w[0] * pts[id[0]] + w[1] * pts[id[1]] +
                                w[2] * pts[id[2]];
                });
    });
    queue.wait_and_throw();
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

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
                sycl::range<3>(static_cast<size_t>(b), static_cast<size_t>(c),
                               static_cast<size_t>(n)),
                [=](sycl::item<3> item) {
                    const int bs_idx = static_cast<int>(item.get_id(0));
                    const int c_idx = static_cast<int>(item.get_id(1));
                    const int pt_idx = static_cast<int>(item.get_id(2));

                    const float g =
                            grad_out[bs_idx * c * n + c_idx * n + pt_idx];
                    const float* const w = weight + bs_idx * n * 3 + pt_idx * 3;
                    const int* const id = idx + bs_idx * n * 3 + pt_idx * 3;
                    float* const gp = grad_points + bs_idx * c * m + c_idx * m;

                    for (int l = 0; l < 3; ++l) {
                        sycl::atomic_ref<
                                float, sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>
                                ref(gp[id[l]]);
                        ref.fetch_add(g * w[l]);
                    }
                });
    });
    queue.wait_and_throw();
}

}  // namespace contrib
}  // namespace ml
}  // namespace open3d
