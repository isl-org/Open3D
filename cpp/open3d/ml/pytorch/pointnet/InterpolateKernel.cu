// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//***************************************************************************************/
//
//    Based on Pointnet2 Library (MIT License):
//    https://github.com/sshaoshuai/Pointnet2.PyTorch
//
//    Copyright (c) 2019 Shaoshuai Shi
//
//    Permission is hereby granted, free of charge, to any person obtaining a
//    copy of this software and associated documentation files (the "Software"),
//    to deal in the Software without restriction, including without limitation
//    the rights to use, copy, modify, merge, publish, distribute, sublicense,
//    and/or sell copies of the Software, and to permit persons to whom the
//    Software is furnished to do so, subject to the following conditions:
//
//    The above copyright notice and this permission notice shall be included in
//    all copies or substantial portions of the Software.
//
//    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//    DEALINGS IN THE SOFTWARE.
//
//***************************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <vector>

#include "ATen/cuda/CUDAContext.h"
#include "open3d/ml/contrib/InterpolatePoints.cuh"
#include "open3d/ml/contrib/cuda_utils.h"
#include "open3d/ml/pytorch/pointnet/InterpolateKernel.h"

using namespace open3d::ml::contrib;

void three_nn_launcher(int b,
                       int n,
                       int m,
                       const float *unknown,
                       const float *known,
                       float *dist2,
                       int *idx) {
    // unknown: (B, N, 3)
    // known: (B, M, 3)
    // output:
    //      dist2: (B, N, 3)
    //      idx: (B, N, 3)

    cudaError_t err;

    auto stream = at::cuda::getCurrentCUDAStream();

    dim3 blocks(DIVUP(n, THREADS_PER_BLOCK),
                b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    three_nn_kernel<<<blocks, threads, 0, stream>>>(b, n, m, unknown, known,
                                                    dist2, idx);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

void three_interpolate_launcher(int b,
                                int c,
                                int m,
                                int n,
                                const float *points,
                                const int *idx,
                                const float *weight,
                                float *out) {
    // points: (B, C, M)
    // idx: (B, N, 3)
    // weight: (B, N, 3)
    // output:
    //      out: (B, C, N)

    cudaError_t err;

    auto stream = at::cuda::getCurrentCUDAStream();

    dim3 blocks(DIVUP(n, THREADS_PER_BLOCK), c,
                b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);
    three_interpolate_kernel<<<blocks, threads, 0, stream>>>(b, c, m, n, points,
                                                             idx, weight, out);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

void three_interpolate_grad_launcher(int b,
                                     int c,
                                     int n,
                                     int m,
                                     const float *grad_out,
                                     const int *idx,
                                     const float *weight,
                                     float *grad_points) {
    // grad_out: (B, C, N)
    // weight: (B, N, 3)
    // output:
    //      grad_points: (B, C, M)

    cudaError_t err;

    auto stream = at::cuda::getCurrentCUDAStream();

    dim3 blocks(DIVUP(n, THREADS_PER_BLOCK), c,
                b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);
    three_interpolate_grad_kernel<<<blocks, threads, 0, stream>>>(
            b, c, n, m, grad_out, idx, weight, grad_points);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
