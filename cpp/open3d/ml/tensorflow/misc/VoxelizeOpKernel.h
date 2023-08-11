// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

//#include "open3d/ml/impl/misc/VoxelPooling.h"
#include "open3d/ml/tensorflow/TensorFlowHelper.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"

/// @cond
// namespace for code that is common for all kernels
namespace voxelize_opkernel {

class OutputAllocator {
public:
    OutputAllocator(tensorflow::OpKernelContext* context) : context(context) {}

    void AllocVoxelCoords(int32_t** ptr, int64_t rows, int64_t cols) {
        using namespace tensorflow;
        *ptr = nullptr;
        Tensor* tensor = 0;
        TensorShape shape({rows, cols});
        OP_REQUIRES_OK(context, context->allocate_output(0, shape, &tensor));
        auto flat_tensor = tensor->flat<int32_t>();
        *ptr = flat_tensor.data();
    }

    void AllocVoxelPointIndices(int64_t** ptr, int64_t num) {
        using namespace tensorflow;
        *ptr = nullptr;
        Tensor* tensor = 0;
        TensorShape shape({num});
        OP_REQUIRES_OK(context, context->allocate_output(1, shape, &tensor));
        auto flat_tensor = tensor->flat<int64>();
        *ptr = (int64_t*)flat_tensor.data();
    }

    void AllocVoxelPointRowSplits(int64_t** ptr, int64_t num) {
        using namespace tensorflow;
        *ptr = nullptr;
        Tensor* tensor = 0;
        TensorShape shape({num});
        OP_REQUIRES_OK(context, context->allocate_output(2, shape, &tensor));
        auto flat_tensor = tensor->flat<int64>();
        *ptr = (int64_t*)flat_tensor.data();
    }

    void AllocVoxelBatchSplits(int64_t** ptr, int64_t num) {
        using namespace tensorflow;
        *ptr = nullptr;
        Tensor* tensor = 0;
        TensorShape shape({num});
        OP_REQUIRES_OK(context, context->allocate_output(3, shape, &tensor));
        auto flat_tensor = tensor->flat<int64>();
        *ptr = (int64_t*)flat_tensor.data();
    }

private:
    tensorflow::OpKernelContext* context;
};

// Base class with common code for the OpKernel implementations
class VoxelizeOpKernel : public tensorflow::OpKernel {
public:
    explicit VoxelizeOpKernel(tensorflow::OpKernelConstruction* construction)
        : OpKernel(construction) {
        OP_REQUIRES_OK(construction,
                       construction->GetAttr("max_points_per_voxel",
                                             &max_points_per_voxel));
        OP_REQUIRES_OK(construction,
                       construction->GetAttr("max_voxels", &max_voxels));
    }

    void Compute(tensorflow::OpKernelContext* context) override {
        using namespace tensorflow;
        const Tensor& points = context->input(0);
        const Tensor& row_splits = context->input(1);
        const Tensor& voxel_size = context->input(2);
        const Tensor& points_range_min = context->input(3);
        const Tensor& points_range_max = context->input(4);

        {
            using namespace open3d::ml::op_util;
            Dim num_points("num_points");
            Dim ndim("ndim");
            CHECK_SHAPE(context, points, num_points, ndim);
            CHECK_SHAPE(context, voxel_size, ndim);
            CHECK_SHAPE(context, points_range_min, ndim);
            CHECK_SHAPE(context, points_range_max, ndim);
            OP_REQUIRES(
                    context, ndim.value() > 0 && ndim.value() < 9,
                    errors::InvalidArgument(
                            "the number of dimensions must be in [1,..,8]"));
        }

        Kernel(context, points, row_splits, voxel_size, points_range_min,
               points_range_max);
    }

    // Function with the device specific code
    virtual void Kernel(tensorflow::OpKernelContext* context,
                        const tensorflow::Tensor& points,
                        const tensorflow::Tensor& row_splits,
                        const tensorflow::Tensor& voxel_size,
                        const tensorflow::Tensor& points_range_min,
                        const tensorflow::Tensor& points_range_max) = 0;

protected:
    tensorflow::int64 max_points_per_voxel;
    tensorflow::int64 max_voxels;
};

}  // namespace voxelize_opkernel
/// @endcond
