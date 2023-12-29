// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/ml/contrib/InterpolatePoints.cuh"

namespace open3d {
namespace ml {
namespace contrib {

__global__ void three_nn_kernel(int b,
                                int n,
                                int m,
                                const float *__restrict__ unknown,
                                const float *__restrict__ known,
                                float *__restrict__ dist2,
                                int *__restrict__ idx) {
    // unknown: (B, N, 3)
    // known: (B, M, 3)
    // output:
    //      dist2: (B, N, 3)
    //      idx: (B, N, 3)

    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || pt_idx >= n) return;

    unknown += bs_idx * n * 3 + pt_idx * 3;
    known += bs_idx * m * 3;
    dist2 += bs_idx * n * 3 + pt_idx * 3;
    idx += bs_idx * n * 3 + pt_idx * 3;

    float ux = unknown[0];
    float uy = unknown[1];
    float uz = unknown[2];

    double best1 = 1e40, best2 = 1e40, best3 = 1e40;
    int besti1 = 0, besti2 = 0, besti3 = 0;
    for (int k = 0; k < m; ++k) {
        float x = known[k * 3 + 0];
        float y = known[k * 3 + 1];
        float z = known[k * 3 + 2];
        float d =
                (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);
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
    dist2[0] = best1;
    dist2[1] = best2;
    dist2[2] = best3;
    idx[0] = besti1;
    idx[1] = besti2;
    idx[2] = besti3;
}

__global__ void three_interpolate_kernel(int b,
                                         int c,
                                         int m,
                                         int n,
                                         const float *__restrict__ points,
                                         const int *__restrict__ idx,
                                         const float *__restrict__ weight,
                                         float *__restrict__ out) {
    // points: (B, C, M)
    // idx: (B, N, 3)
    // weight: (B, N, 3)
    // output:
    //      out: (B, C, N)

    int bs_idx = blockIdx.z;
    int c_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (bs_idx >= b || c_idx >= c || pt_idx >= n) return;

    weight += bs_idx * n * 3 + pt_idx * 3;
    points += bs_idx * c * m + c_idx * m;
    idx += bs_idx * n * 3 + pt_idx * 3;
    out += bs_idx * c * n + c_idx * n;

    out[pt_idx] = weight[0] * points[idx[0]] + weight[1] * points[idx[1]] +
                  weight[2] * points[idx[2]];
}

__global__ void three_interpolate_grad_kernel(
        int b,
        int c,
        int n,
        int m,
        const float *__restrict__ grad_out,
        const int *__restrict__ idx,
        const float *__restrict__ weight,
        float *__restrict__ grad_points) {
    // grad_out: (B, C, N)
    // weight: (B, N, 3)
    // output:
    //      grad_points: (B, C, M)

    int bs_idx = blockIdx.z;
    int c_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (bs_idx >= b || c_idx >= c || pt_idx >= n) return;

    grad_out += bs_idx * c * n + c_idx * n + pt_idx;
    weight += bs_idx * n * 3 + pt_idx * 3;
    grad_points += bs_idx * c * m + c_idx * m;
    idx += bs_idx * n * 3 + pt_idx * 3;

    atomicAdd(grad_points + idx[0], grad_out[0] * weight[0]);
    atomicAdd(grad_points + idx[1], grad_out[0] * weight[1]);
    atomicAdd(grad_points + idx[2], grad_out[0] * weight[2]);
}

}  // namespace contrib
}  // namespace ml
}  // namespace open3d
