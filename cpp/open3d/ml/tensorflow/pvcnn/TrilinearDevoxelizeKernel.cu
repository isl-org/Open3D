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
#include "TrilinearDevoxelizeKernel.h"
#include "open3d/ml/Helper.h"
#include "open3d/ml/contrib/TrilinearDevoxelize.cuh"
#include "open3d/ml/contrib/cuda_utils.h"

using namespace open3d;
using namespace open3d::ml;
using namespace open3d::ml::contrib;
using namespace tensorflow;

class TrilinearDevoxelizeOpKernelCUDA : public TrilinearDevoxelizeOpKernel {
public:
    explicit TrilinearDevoxelizeOpKernelCUDA(OpKernelConstruction* context)
        : TrilinearDevoxelizeOpKernel(context) {}

    void Kernel(tensorflow::OpKernelContext* context,
                int b,
                int c,
                int n,
                int r,
                int r2,
                int r3,
                bool training,
                const float* coords,
                const float* feat,
                int* inds,
                float* wgts,
                float* outs) {
        auto stream = context->eigen_gpu_device().stream();

        cudaError_t err;

        TrilinearDevoxelizeKernel<<<b, OptNumThreads(n), 0, stream>>>(
                b, c, n, r, r2, r3, training, coords, feat, inds, wgts, outs);

        err = cudaGetLastError();
        if (cudaSuccess != err) {
            fprintf(stderr, "CUDA kernel failed : %s\n",
                    cudaGetErrorString(err));
            exit(-1);
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("Open3DTrilinearDevoxelize").Device(DEVICE_GPU),
                        TrilinearDevoxelizeOpKernelCUDA)

class TrilinearDevoxelizeGradOpKernelCUDA
    : public TrilinearDevoxelizeGradOpKernel {
public:
    explicit TrilinearDevoxelizeGradOpKernelCUDA(OpKernelConstruction* context)
        : TrilinearDevoxelizeGradOpKernel(context) {}

    void Kernel(tensorflow::OpKernelContext* context,
                int b,
                int c,
                int n,
                int r3,
                const int* inds,
                const float* wgts,
                const float* grad_y,
                float* grad_x) {
        auto stream = context->eigen_gpu_device().stream();

        cudaError_t err;

        TrilinearDevoxelizeGradKernel<<<b, OptNumThreads(n), 0, stream>>>(
                b, c, n, r3, inds, wgts, grad_y, grad_x);

        err = cudaGetLastError();
        if (cudaSuccess != err) {
            fprintf(stderr, "CUDA kernel failed : %s\n",
                    cudaGetErrorString(err));
            exit(-1);
        }
    }
};

REGISTER_KERNEL_BUILDER(
        Name("Open3DTrilinearDevoxelizeGrad").Device(DEVICE_GPU),
        TrilinearDevoxelizeGradOpKernelCUDA)
