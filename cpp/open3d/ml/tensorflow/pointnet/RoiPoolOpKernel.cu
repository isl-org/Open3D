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
