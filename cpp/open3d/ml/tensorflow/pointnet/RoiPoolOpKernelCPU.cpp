// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include <cstring>

#include "RoiPoolOpKernel.h"
#include "open3d/ml/contrib/RoiPoolKernel.h"

using namespace open3d::ml::contrib;
using namespace tensorflow;

class RoiPoolOpKernelCPU : public RoiPoolOpKernel {
public:
    explicit RoiPoolOpKernelCPU(OpKernelConstruction *construction)
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
        memset(pooled_features, 0,
               batch_size * boxes_num * sampled_pts_num * (3 + feature_in_len) *
                       sizeof(float));
        memset(pooled_empty_flag, 0, batch_size * boxes_num * sizeof(int));

        roipool3dLauncherCPU(batch_size, pts_num, boxes_num, feature_in_len,
                             sampled_pts_num, xyz, boxes3d, pts_feature,
                             pooled_features, pooled_empty_flag);
    }
};

REGISTER_KERNEL_BUILDER(Name("Open3DRoiPool").Device(DEVICE_CPU),
                        RoiPoolOpKernelCPU);
