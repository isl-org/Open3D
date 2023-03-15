// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#define EIGEN_USE_GPU
#include "InterpolateOpKernel.h"
#include "open3d/ml/Helper.h"
#include "open3d/ml/contrib/InterpolatePoints.cuh"
#include "open3d/ml/contrib/cuda_utils.h"

using namespace open3d;
using namespace open3d::ml;
using namespace open3d::ml::contrib;
using namespace tensorflow;

class ThreeNNOpKernelCUDA : public ThreeNNOpKernel {
public:
    explicit ThreeNNOpKernelCUDA(OpKernelConstruction *construction)
        : ThreeNNOpKernel(construction) {}

    void Kernel(tensorflow::OpKernelContext *context,
                int b,
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

        auto stream = context->eigen_gpu_device().stream();

        cudaError_t err;

        dim3 blocks(DIVUP(n, THREADS_PER_BLOCK),
                    b);  // blockIdx.x(col), blockIdx.y(row)
        dim3 threads(THREADS_PER_BLOCK);

        three_nn_kernel<<<blocks, threads, 0, stream>>>(b, n, m, unknown, known,
                                                        dist2, idx);

        err = cudaGetLastError();
        if (cudaSuccess != err) {
            fprintf(stderr, "CUDA kernel failed : %s\n",
                    cudaGetErrorString(err));
            exit(-1);
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("Open3DThreeNN").Device(DEVICE_GPU),
                        ThreeNNOpKernelCUDA);

class ThreeInterpolateOpKernelCUDA : public ThreeInterpolateOpKernel {
public:
    explicit ThreeInterpolateOpKernelCUDA(OpKernelConstruction *construction)
        : ThreeInterpolateOpKernel(construction) {}

    void Kernel(tensorflow::OpKernelContext *context,
                int b,
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

        auto stream = context->eigen_gpu_device().stream();

        cudaError_t err;
        dim3 blocks(DIVUP(n, THREADS_PER_BLOCK), c,
                    b);  // blockIdx.x(col), blockIdx.y(row)
        dim3 threads(THREADS_PER_BLOCK);
        three_interpolate_kernel<<<blocks, threads, 0, stream>>>(
                b, c, m, n, points, idx, weight, out);

        err = cudaGetLastError();
        if (cudaSuccess != err) {
            fprintf(stderr, "CUDA kernel failed : %s\n",
                    cudaGetErrorString(err));
            exit(-1);
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("Open3DThreeInterpolate").Device(DEVICE_GPU),
                        ThreeInterpolateOpKernelCUDA);

class ThreeInterpolateGradOpKernelCUDA : public ThreeInterpolateGradOpKernel {
public:
    explicit ThreeInterpolateGradOpKernelCUDA(
            OpKernelConstruction *construction)
        : ThreeInterpolateGradOpKernel(construction) {}

    void Kernel(tensorflow::OpKernelContext *context,
                int b,
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

        auto stream = context->eigen_gpu_device().stream();

        cudaError_t err;

        dim3 blocks(DIVUP(n, THREADS_PER_BLOCK), c,
                    b);  // blockIdx.x(col), blockIdx.y(row)
        dim3 threads(THREADS_PER_BLOCK);
        three_interpolate_grad_kernel<<<blocks, threads, 0, stream>>>(
                b, c, n, m, grad_out, idx, weight, grad_points);

        err = cudaGetLastError();
        if (cudaSuccess != err) {
            fprintf(stderr, "CUDA kernel failed : %s\n",
                    cudaGetErrorString(err));
            exit(-1);
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("Open3DThreeInterpolateGrad").Device(DEVICE_GPU),
                        ThreeInterpolateGradOpKernelCUDA);
