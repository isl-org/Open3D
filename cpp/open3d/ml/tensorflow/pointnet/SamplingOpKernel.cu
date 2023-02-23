// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#define EIGEN_USE_GPU
#include "SamplingOpKernel.h"
#include "open3d/ml/Helper.h"
#include "open3d/ml/contrib/PointSampling.cuh"
#include "open3d/ml/contrib/cuda_utils.h"

using namespace open3d;
using namespace open3d::ml;
using namespace open3d::ml::contrib;
using namespace tensorflow;

class FurthestPointSamplingOpKernelCUDA : public FurthestPointSamplingOpKernel {
public:
    explicit FurthestPointSamplingOpKernelCUDA(
            OpKernelConstruction *construction)
        : FurthestPointSamplingOpKernel(construction) {}

    void Kernel(tensorflow::OpKernelContext *context,
                int b,
                int n,
                int m,
                const float *dataset,
                float *temp,
                int *idxs) {
        // dataset: (B, N, 3)
        // tmp: (B, N)
        // output:
        //      idx: (B, M)

        // fill with big value
        cudaMemset(temp, 80, b * n * sizeof(float));

        auto stream = context->eigen_gpu_device().stream();

        cudaError_t err;

        unsigned int n_threads = OptNumThreads(n);

        switch (n_threads) {
            case 1024:
                furthest_point_sampling_kernel<1024>
                        <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp,
                                                      idxs);
                break;
            case 512:
                furthest_point_sampling_kernel<512>
                        <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp,
                                                      idxs);
                break;
            case 256:
                furthest_point_sampling_kernel<256>
                        <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp,
                                                      idxs);
                break;
            case 128:
                furthest_point_sampling_kernel<128>
                        <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp,
                                                      idxs);
                break;
            case 64:
                furthest_point_sampling_kernel<64><<<b, n_threads, 0, stream>>>(
                        b, n, m, dataset, temp, idxs);
                break;
            case 32:
                furthest_point_sampling_kernel<32><<<b, n_threads, 0, stream>>>(
                        b, n, m, dataset, temp, idxs);
                break;
            case 16:
                furthest_point_sampling_kernel<16><<<b, n_threads, 0, stream>>>(
                        b, n, m, dataset, temp, idxs);
                break;
            case 8:
                furthest_point_sampling_kernel<8><<<b, n_threads, 0, stream>>>(
                        b, n, m, dataset, temp, idxs);
                break;
            case 4:
                furthest_point_sampling_kernel<4><<<b, n_threads, 0, stream>>>(
                        b, n, m, dataset, temp, idxs);
                break;
            case 2:
                furthest_point_sampling_kernel<2><<<b, n_threads, 0, stream>>>(
                        b, n, m, dataset, temp, idxs);
                break;
            case 1:
                furthest_point_sampling_kernel<1><<<b, n_threads, 0, stream>>>(
                        b, n, m, dataset, temp, idxs);
                break;
            default:
                furthest_point_sampling_kernel<512>
                        <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp,
                                                      idxs);
        }

        err = cudaGetLastError();
        if (cudaSuccess != err) {
            fprintf(stderr, "CUDA kernel failed : %s\n",
                    cudaGetErrorString(err));
            exit(-1);
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("Open3DFurthestPointSampling").Device(DEVICE_GPU),
                        FurthestPointSamplingOpKernelCUDA);
