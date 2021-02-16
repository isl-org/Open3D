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

#include "ATen/cuda/CUDAContext.h"
#include "open3d/ml/pytorch/pointnet/BallQueryKernel.h"
#include "open3d/ml/pytorch/pointnet/cuda_utils.h"

__global__ void ball_query_kernel(int b,
                                  int n,
                                  int m,
                                  float radius,
                                  int nsample,
                                  const float *__restrict__ new_xyz,
                                  const float *__restrict__ xyz,
                                  int *__restrict__ idx) {
    // new_xyz: (B, M, 3)
    // xyz: (B, N, 3)
    // output:
    //      idx: (B, M, nsample)
    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || pt_idx >= m) return;

    new_xyz += bs_idx * m * 3 + pt_idx * 3;
    xyz += bs_idx * n * 3;
    idx += bs_idx * m * nsample + pt_idx * nsample;

    float radius2 = radius * radius;
    float new_x = new_xyz[0];
    float new_y = new_xyz[1];
    float new_z = new_xyz[2];

    int cnt = 0;
    for (int k = 0; k < n; ++k) {
        float x = xyz[k * 3 + 0];
        float y = xyz[k * 3 + 1];
        float z = xyz[k * 3 + 2];
        float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
                   (new_z - z) * (new_z - z);
        if (d2 < radius2) {
            if (cnt == 0) {
                for (int l = 0; l < nsample; ++l) {
                    idx[l] = k;
                }
            }
            idx[cnt] = k;
            ++cnt;
            if (cnt >= nsample) break;
        }
    }
}

void ball_query_launcher(int b,
                         int n,
                         int m,
                         float radius,
                         int nsample,
                         const float *new_xyz,
                         const float *xyz,
                         int *idx) {
    // new_xyz: (B, M, 3)
    // xyz: (B, N, 3)
    // output:
    //      idx: (B, M, nsample)

    cudaError_t err;

    auto stream = at::cuda::getCurrentCUDAStream();

    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK),
                b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    ball_query_kernel<<<blocks, threads, 0, stream>>>(b, n, m, radius, nsample,
                                                      new_xyz, xyz, idx);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
