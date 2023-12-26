// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "VoxelPoolingOpKernel.h"

#include "open3d/ml/impl/misc/VoxelPooling.h"

using namespace open3d::ml::impl;
using namespace voxel_pooling_opkernel;
using namespace tensorflow;

template <class TReal, class TFeat>
class VoxelPoolingOpKernelCPU : public VoxelPoolingOpKernel {
public:
    explicit VoxelPoolingOpKernelCPU(OpKernelConstruction* construction)
        : VoxelPoolingOpKernel(construction) {}

    void Kernel(tensorflow::OpKernelContext* context,
                const tensorflow::Tensor& positions,
                const tensorflow::Tensor& features,
                const tensorflow::Tensor& voxel_size) {
        OutputAllocator<TReal, TFeat> output_allocator(context);

        if (debug) {
            std::string err;
            OP_REQUIRES(context,
                        CheckVoxelSize(err, positions.shape().dim_size(0),
                                       positions.flat<TReal>().data(),
                                       voxel_size.scalar<TReal>()()),
                        errors::InvalidArgument(err));
        }

        VoxelPooling<TReal, TFeat>(
                positions.shape().dim_size(0), positions.flat<TReal>().data(),
                features.shape().dim_size(1), features.flat<TFeat>().data(),
                voxel_size.scalar<TReal>()(), output_allocator, position_fn,
                feature_fn);
    }
};

#define REG_KB(type, typefeat)                                          \
    REGISTER_KERNEL_BUILDER(Name("Open3DVoxelPooling")                  \
                                    .Device(DEVICE_CPU)                 \
                                    .TypeConstraint<type>("TReal")      \
                                    .TypeConstraint<typefeat>("TFeat"), \
                            VoxelPoolingOpKernelCPU<type, typefeat>);
REG_KB(float, float)
REG_KB(float, int)
REG_KB(float, int64)
REG_KB(float, double)
REG_KB(double, float)
REG_KB(double, int)
REG_KB(double, int64)
REG_KB(double, double)
#undef REG_KB
