// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#define EIGEN_USE_GPU
#include "RoiPoolOpKernel.h"
#include "open3d/ml/Helper.h"
#include "open3d/ml/contrib/RoiPoolKernel.h"

using namespace open3d;
using namespace open3d::ml;
using namespace open3d::ml::contrib;
using namespace tensorflow;

class RoiPoolOpKernelCUDA : public RoiPoolOpKernel {
public:
    explicit RoiPoolOpKernelCUDA(OpKernelConstruction *construction)
        : RoiPoolOpKernel(construction) {}

    void Kernel(tensorflow::OpKernelContext *context,
                int batch_size,
                int pts_num,
                int boxes_num,
                int feature_in_len,
                int sampled_pts_num,
                const float *xyz,
                const float *boxes3d,
                const float *pts_feature,
                float *pooled_features,
                int *pooled_empty_flag) {
        cudaError_t err;

        cudaMemset(pooled_features, 0,
                   batch_size * boxes_num * sampled_pts_num *
                           (3 + feature_in_len) * sizeof(float));
        cudaMemset(pooled_empty_flag, 0, batch_size * boxes_num * sizeof(int));

        roipool3dLauncher(batch_size, pts_num, boxes_num, feature_in_len,
                          sampled_pts_num, xyz, boxes3d, pts_feature,
                          pooled_features, pooled_empty_flag);

        // cudaDeviceSynchronize();  // for using printf in kernel function
        err = cudaGetLastError();
        if (cudaSuccess != err) {
            fprintf(stderr, "CUDA kernel failed : %s\n",
                    cudaGetErrorString(err));
            exit(-1);
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("Open3DRoiPool").Device(DEVICE_GPU),
                        RoiPoolOpKernelCUDA);
