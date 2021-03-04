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

#include <stdio.h>
#include <stdlib.h>

#include "ATen/cuda/CUDAContext.h"
#include "open3d/ml/contrib/GroupPoints.cuh"
#include "open3d/ml/contrib/cuda_utils.h"
#include "open3d/ml/pytorch/pointnet/GroupPointsKernel.h"

using namespace open3d::ml::contrib;

void group_points_grad_launcher(int b,
                                int c,
                                int n,
                                int npoints,
                                int nsample,
                                const float *grad_out,
                                const int *idx,
                                float *grad_points) {
    // grad_out: (B, C, npoints, nsample)
    // idx: (B, npoints, nsample)
    // output:
    //      grad_points: (B, C, N)
    cudaError_t err;

    auto stream = at::cuda::getCurrentCUDAStream();

    dim3 blocks(DIVUP(npoints * nsample, THREADS_PER_BLOCK), c,
                b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    group_points_grad_kernel<<<blocks, threads, 0, stream>>>(
            b, c, n, npoints, nsample, grad_out, idx, grad_points);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

void group_points_launcher(int b,
                           int c,
                           int n,
                           int npoints,
                           int nsample,
                           const float *points,
                           const int *idx,
                           float *out) {
    // points: (B, C, N)
    // idx: (B, npoints, nsample)
    // output:
    //      out: (B, C, npoints, nsample)
    cudaError_t err;

    auto stream = at::cuda::getCurrentCUDAStream();

    dim3 blocks(DIVUP(npoints * nsample, THREADS_PER_BLOCK), c,
                b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    group_points_kernel<<<blocks, threads, 0, stream>>>(
            b, c, n, npoints, nsample, points, idx, out);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
