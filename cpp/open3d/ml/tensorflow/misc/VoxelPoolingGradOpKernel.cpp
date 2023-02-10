// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "VoxelPoolingGradOpKernel.h"

#include "open3d/ml/impl/misc/VoxelPooling.h"

using namespace open3d::ml::impl;
using namespace voxel_pooling_opkernel;
using namespace tensorflow;

template <class TReal, class TFeat>
class VoxelPoolingGradOpKernelCPU : public VoxelPoolingGradOpKernel {
public:
    explicit VoxelPoolingGradOpKernelCPU(OpKernelConstruction* construction)
        : VoxelPoolingGradOpKernel(construction) {}

    void Kernel(tensorflow::OpKernelContext* context,
                tensorflow::Tensor& features_backprop,
                const tensorflow::Tensor& positions,
                const tensorflow::Tensor& features,
                const tensorflow::Tensor& pooled_positions,
                const tensorflow::Tensor& pooled_features_gradient,
                const tensorflow::Tensor& voxel_size) {
        VoxelPoolingBackprop<TReal, TFeat>(
                features_backprop.flat<TFeat>().data(),
                positions.shape().dim_size(0), positions.flat<TReal>().data(),
                features.shape().dim_size(1), features.flat<TFeat>().data(),
                pooled_positions.shape().dim_size(0),
                pooled_positions.flat<TReal>().data(),
                pooled_features_gradient.flat<TFeat>().data(),
                voxel_size.scalar<TReal>()(), position_fn, feature_fn);
    }
};

#define REG_KB(type, typefeat)                                          \
    REGISTER_KERNEL_BUILDER(Name("Open3DVoxelPoolingGrad")              \
                                    .Device(DEVICE_CPU)                 \
                                    .TypeConstraint<type>("TReal")      \
                                    .TypeConstraint<typefeat>("TFeat"), \
                            VoxelPoolingGradOpKernelCPU<type, typefeat>);
REG_KB(float, float)
REG_KB(float, double)
REG_KB(double, float)
REG_KB(double, double)
// gradient computation is not supported for integer feature types by tensorflow
#undef REG_KB
