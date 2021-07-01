// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------
//

#define EIGEN_USE_GPU
#include "BallQueryOpKernel.h"
#include "open3d/ml/Helper.h"
#include "open3d/ml/contrib/BallQuery.cuh"
#include "open3d/ml/contrib/cuda_utils.h"

using namespace open3d;
using namespace open3d::ml;
using namespace open3d::ml::contrib;
using namespace tensorflow;

class BallQueryOpKernelCUDA : public BallQueryOpKernel {
public:
    explicit BallQueryOpKernelCUDA(OpKernelConstruction *construction)
        : BallQueryOpKernel(construction) {}

    void Kernel(tensorflow::OpKernelContext *context,
                int b,
                int n,
                int m,
                float radius,
                int nsample,
                const float *new_xyz,
                const float *xyz,
                int *idx) {
        // dataset: (B, N, 3)
        // tmp: (B, N)
        // output:
        //      idx: (B, M)

        auto stream = context->eigen_gpu_device().stream();

        cudaError_t err;

        dim3 blocks(DIVUP(m, THREADS_PER_BLOCK),
                    b);  // blockIdx.x(col), blockIdx.y(row)
        dim3 threads(THREADS_PER_BLOCK);

        ball_query_kernel<<<blocks, threads, 0, stream>>>(
                b, n, m, radius, nsample, new_xyz, xyz, idx);
        // cudaDeviceSynchronize();  // for using printf in kernel function
        err = cudaGetLastError();
        if (cudaSuccess != err) {
            fprintf(stderr, "CUDA kernel failed : %s\n",
                    cudaGetErrorString(err));
            exit(-1);
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("Open3DBallQuery").Device(DEVICE_GPU),
                        BallQueryOpKernelCUDA);
