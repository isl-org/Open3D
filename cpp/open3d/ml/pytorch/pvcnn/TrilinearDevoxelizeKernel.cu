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
#include "open3d/ml/contrib/TrilinearDevoxelize.cuh"
#include "open3d/ml/contrib/cuda_utils.h"
#include "open3d/ml/pytorch/pointnet/TrilinearDevoxelizeKernel.h"

using namespace open3d::ml::contrib;

void trilinear_devoxelize(int b,
                          int c,
                          int n,
                          int r,
                          int r2,
                          int r3,
                          bool training,
                          const float *coords,
                          const float *feat,
                          int *inds,
                          float *wgts,
                          float *outs) {
    cudaError_t err;

    auto stream = at::cuda::getCurrentCUDAStream();

    trilinear_devoxelize_kernel<<<b, optimal_num_threads(n), 0, stream>>>(
            b, c, n, r, r2, r3, training, coords, feat, inds, wgts, outs);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

void trilinear_devoxelize_grad(int b,
                               int c,
                               int n,
                               int r3,
                               const int *inds,
                               const float *wgts,
                               const float *grad_y,
                               float *grad_x) {
    cudaError_t err;

    auto stream = at::cuda::getCurrentCUDAStream();

    trilinear_devoxelize_grad_kernel<<<b, optimal_num_threads(n), 0, stream>>>(
            b, c, n, r3, inds, wgts, grad_y, grad_x);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
