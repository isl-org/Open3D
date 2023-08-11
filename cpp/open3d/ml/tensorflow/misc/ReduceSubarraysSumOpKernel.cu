// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#define EIGEN_USE_GPU
#include "ReduceSubarraysSumOpKernel.h"
#include "open3d/ml/impl/misc/ReduceSubarraysSum.cuh"

using namespace open3d::ml::impl;
using namespace reduce_subarrays_sum_opkernel;
using namespace tensorflow;

template <class T>
class ReduceSubarraysSumOpKernelCUDA : public ReduceSubarraysSumOpKernel {
public:
    explicit ReduceSubarraysSumOpKernelCUDA(OpKernelConstruction* construction)
        : ReduceSubarraysSumOpKernel(construction) {}

    void Kernel(OpKernelContext* context,
                const tensorflow::Tensor& values,
                const tensorflow::Tensor& row_splits,
                tensorflow::Tensor& sums) {
        auto device = context->eigen_gpu_device();

        ReduceSubarraysSumCUDA(device.stream(), values.flat<T>().data(),
                               values.shape().dim_size(0),
                               (int64_t*)row_splits.flat<int64>().data(),
                               row_splits.shape().dim_size(0) - 1,
                               sums.flat<T>().data());
    }
};

#define REG_KB(type)                                            \
    REGISTER_KERNEL_BUILDER(Name("Open3DReduceSubarraysSum")    \
                                    .Device(DEVICE_GPU)         \
                                    .TypeConstraint<type>("T"), \
                            ReduceSubarraysSumOpKernelCUDA<type>);
REG_KB(int32_t)
REG_KB(int64)
REG_KB(float)
REG_KB(double)
#undef REG_KB
