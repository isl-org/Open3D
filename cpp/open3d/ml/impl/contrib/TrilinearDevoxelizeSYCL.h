// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// SYCL implementation of TrilinearDevoxelize(Grad) — ports
// TrilinearDevoxelize.cu. Forward is a per-point trilinear gather (no shared
// state between points, so we use a flat 2-D parallel_for over (batch,
// point) rather than the CUDA grid-stride-over-one-block-per-batch layout).
// Backward scatters into the voxel grid via atomic_ref::fetch_add (proven
// pattern from InvertNeighborsList / InterpolatePoints).

#pragma once

#include <sycl/sycl.hpp>

namespace open3d {
namespace ml {
namespace contrib {

/// Trilinear devoxelization: for each of the n query points per batch,
/// gathers and interpolates the 8 surrounding voxels of the [c, r, r, r]
/// feature grid. If is_training, also records the 8 voxel indices and
/// weights (needed by the backward pass).
inline void TrilinearDevoxelizeSYCL(sycl::queue& queue,
                                    int b,
                                    int c,
                                    int n,
                                    int r,
                                    int r2,
                                    int r3,
                                    bool is_training,
                                    const float* const coords,
                                    const float* const feat,
                                    int* const inds,
                                    float* const wgts,
                                    float* const outs) {
    if (b <= 0 || n <= 0) return;

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
                sycl::range<2>(static_cast<size_t>(b), static_cast<size_t>(n)),
                [=](sycl::item<2> item) {
                    const int batch_index = static_cast<int>(item.get_id(0));
                    const int i = static_cast<int>(item.get_id(1));

                    const float* const crd = coords + batch_index * n * 3;
                    int* const ind = inds + batch_index * n * 8;
                    float* const wgt = wgts + batch_index * n * 8;
                    const float* const ft = feat + batch_index * c * r3;
                    float* const out = outs + batch_index * c * n;

                    const float x = crd[i];
                    const float y = crd[i + n];
                    const float z = crd[i + n + n];
                    const float x_lo_f = sycl::floor(x);
                    const float y_lo_f = sycl::floor(y);
                    const float z_lo_f = sycl::floor(z);

                    const float x_d_1 = x - x_lo_f;
                    const float y_d_1 = y - y_lo_f;
                    const float z_d_1 = z - z_lo_f;
                    const float x_d_0 = 1.0f - x_d_1;
                    const float y_d_0 = 1.0f - y_d_1;
                    const float z_d_0 = 1.0f - z_d_1;

                    const float wgt000 = x_d_0 * y_d_0 * z_d_0;
                    const float wgt001 = x_d_0 * y_d_0 * z_d_1;
                    const float wgt010 = x_d_0 * y_d_1 * z_d_0;
                    const float wgt011 = x_d_0 * y_d_1 * z_d_1;
                    const float wgt100 = x_d_1 * y_d_0 * z_d_0;
                    const float wgt101 = x_d_1 * y_d_0 * z_d_1;
                    const float wgt110 = x_d_1 * y_d_1 * z_d_0;
                    const float wgt111 = x_d_1 * y_d_1 * z_d_1;

                    const int x_lo = static_cast<int>(x_lo_f);
                    const int y_lo = static_cast<int>(y_lo_f);
                    const int z_lo = static_cast<int>(z_lo_f);
                    const int x_hi = (x_d_1 > 0) ? -1 : 0;
                    const int y_hi = (y_d_1 > 0) ? -1 : 0;
                    const int z_hi = (z_d_1 > 0) ? 1 : 0;

                    const int idx000 = x_lo * r2 + y_lo * r + z_lo;
                    const int idx001 = idx000 + z_hi;
                    const int idx010 = idx000 + (y_hi & r);
                    const int idx011 = idx010 + z_hi;
                    const int idx100 = idx000 + (x_hi & r2);
                    const int idx101 = idx100 + z_hi;
                    const int idx110 = idx100 + (y_hi & r);
                    const int idx111 = idx110 + z_hi;

                    if (is_training) {
                        wgt[i] = wgt000;
                        wgt[i + n] = wgt001;
                        wgt[i + n * 2] = wgt010;
                        wgt[i + n * 3] = wgt011;
                        wgt[i + n * 4] = wgt100;
                        wgt[i + n * 5] = wgt101;
                        wgt[i + n * 6] = wgt110;
                        wgt[i + n * 7] = wgt111;
                        ind[i] = idx000;
                        ind[i + n] = idx001;
                        ind[i + n * 2] = idx010;
                        ind[i + n * 3] = idx011;
                        ind[i + n * 4] = idx100;
                        ind[i + n * 5] = idx101;
                        ind[i + n * 6] = idx110;
                        ind[i + n * 7] = idx111;
                    }

                    for (int j = 0; j < c; ++j) {
                        const int jr3 = j * r3;
                        out[j * n + i] = wgt000 * ft[jr3 + idx000] +
                                         wgt001 * ft[jr3 + idx001] +
                                         wgt010 * ft[jr3 + idx010] +
                                         wgt011 * ft[jr3 + idx011] +
                                         wgt100 * ft[jr3 + idx100] +
                                         wgt101 * ft[jr3 + idx101] +
                                         wgt110 * ft[jr3 + idx110] +
                                         wgt111 * ft[jr3 + idx111];
                    }
                });
    });
    queue.wait_and_throw();
}

/// Gradient of TrilinearDevoxelizeSYCL w.r.t. the feature grid. grad_x must
/// be zero-initialized by the caller before this call (uses atomic scatter).
inline void TrilinearDevoxelizeGradSYCL(sycl::queue& queue,
                                        int b,
                                        int c,
                                        int n,
                                        int r3,
                                        const int* const inds,
                                        const float* const wgts,
                                        const float* const grad_y,
                                        float* const grad_x) {
    if (b <= 0 || n <= 0) return;

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
                sycl::range<2>(static_cast<size_t>(b), static_cast<size_t>(n)),
                [=](sycl::item<2> item) {
                    const int batch_index = static_cast<int>(item.get_id(0));
                    const int i = static_cast<int>(item.get_id(1));

                    const int* const ind = inds + batch_index * n * 8;
                    const float* const wgt = wgts + batch_index * n * 8;
                    float* const gx = grad_x + batch_index * c * r3;
                    const float* const gy = grad_y + batch_index * c * n;

                    const int idx[8] = {ind[i],         ind[i + n],
                                        ind[i + n * 2], ind[i + n * 3],
                                        ind[i + n * 4], ind[i + n * 5],
                                        ind[i + n * 6], ind[i + n * 7]};
                    const float w[8] = {wgt[i],         wgt[i + n],
                                        wgt[i + n * 2], wgt[i + n * 3],
                                        wgt[i + n * 4], wgt[i + n * 5],
                                        wgt[i + n * 6], wgt[i + n * 7]};

                    for (int j = 0; j < c; ++j) {
                        const int jr3 = j * r3;
                        const float g = gy[j * n + i];
                        for (int l = 0; l < 8; ++l) {
                            sycl::atomic_ref<
                                    float, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                                    ref(gx[jr3 + idx[l]]);
                            ref.fetch_add(w[l] * g);
                        }
                    }
                });
    });
    queue.wait_and_throw();
}

}  // namespace contrib
}  // namespace ml
}  // namespace open3d
