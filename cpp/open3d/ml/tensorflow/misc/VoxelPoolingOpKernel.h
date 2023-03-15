// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/ml/impl/misc/VoxelPooling.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"

/// @cond
// namespace for code that is common for all kernels
namespace voxel_pooling_opkernel {

template <class TReal, class TFeat>
class OutputAllocator {
public:
    OutputAllocator(tensorflow::OpKernelContext* context) : context(context) {}

    void AllocPooledPositions(TReal** ptr, size_t num) {
        using namespace tensorflow;
        *ptr = nullptr;
        Tensor* tensor = 0;
        TensorShape shape({int64_t(num), 3});
        OP_REQUIRES_OK(context, context->allocate_output(0, shape, &tensor));
        auto flat_tensor = tensor->flat<TReal>();
        *ptr = flat_tensor.data();
    }

    void AllocPooledFeatures(TFeat** ptr, size_t num, int channels) {
        using namespace tensorflow;
        *ptr = nullptr;
        Tensor* tensor = 0;
        TensorShape shape({int64_t(num), channels});
        OP_REQUIRES_OK(context, context->allocate_output(1, shape, &tensor));
        auto flat_tensor = tensor->flat<TFeat>();
        *ptr = flat_tensor.data();
    }

private:
    tensorflow::OpKernelContext* context;
};

// Base class with common code for the OpKernel implementations
class VoxelPoolingOpKernel : public tensorflow::OpKernel {
public:
    explicit VoxelPoolingOpKernel(
            tensorflow::OpKernelConstruction* construction)
        : OpKernel(construction) {
        using namespace tensorflow;
        using namespace open3d::ml::impl;
        std::string pos_fn_str;
        OP_REQUIRES_OK(construction,
                       construction->GetAttr("position_fn", &pos_fn_str));

        if (pos_fn_str == "average")
            position_fn = AVERAGE;
        else if (pos_fn_str == "nearest_neighbor")
            position_fn = NEAREST_NEIGHBOR;
        else
            position_fn = CENTER;

        std::string feat_fn_str;
        OP_REQUIRES_OK(construction,
                       construction->GetAttr("feature_fn", &feat_fn_str));

        if (feat_fn_str == "average")
            feature_fn = AVERAGE;
        else if (feat_fn_str == "nearest_neighbor")
            feature_fn = NEAREST_NEIGHBOR;
        else
            feature_fn = MAX;

        OP_REQUIRES_OK(construction, construction->GetAttr("debug", &debug));
    }

    void Compute(tensorflow::OpKernelContext* context) override {
        using namespace tensorflow;
        using namespace open3d::ml::impl;
        const Tensor& positions = context->input(0);
        OP_REQUIRES(
                context, positions.shape().dims() == 2,
                errors::InvalidArgument("positions must be a rank 2 tensor"));

        const Tensor& features = context->input(1);
        OP_REQUIRES(
                context, features.shape().dims() == 2,
                errors::InvalidArgument("features must be a rank 2 tensor"));

        const Tensor& voxel_size = context->input(2);
        OP_REQUIRES(
                context, TensorShapeUtils::IsScalar(voxel_size.shape()),
                errors::InvalidArgument("voxel_size must be a scalar, but is ",
                                        voxel_size.shape().DebugString()));

        Kernel(context, positions, features, voxel_size);
    }

    // Function with the device specific code
    virtual void Kernel(tensorflow::OpKernelContext* context,
                        const tensorflow::Tensor& positions,
                        const tensorflow::Tensor& features,
                        const tensorflow::Tensor& voxel_size) = 0;

protected:
    open3d::ml::impl::AccumulationFn position_fn;
    open3d::ml::impl::AccumulationFn feature_fn;
    bool debug;
};

}  // namespace voxel_pooling_opkernel
/// @endcond
