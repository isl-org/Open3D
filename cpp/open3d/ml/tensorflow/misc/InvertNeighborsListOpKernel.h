// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "../TensorFlowHelper.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"

/// @cond
// namespace for code that is common for all kernels
namespace invert_neighbors_list_opkernel {

// Base class with common code for the OpKernel implementations
class InvertNeighborsListOpKernel : public tensorflow::OpKernel {
public:
    explicit InvertNeighborsListOpKernel(
            tensorflow::OpKernelConstruction* construction)
        : OpKernel(construction) {}

    void Compute(tensorflow::OpKernelContext* context) override {
        using namespace tensorflow;
        static_assert(sizeof(int64) == sizeof(int64_t),
                      "int64 type is not compatible");

        const Tensor& num_points_tensor = context->input(0);
        OP_REQUIRES(context,
                    TensorShapeUtils::IsScalar(num_points_tensor.shape()),
                    errors::InvalidArgument(
                            "num_points must be scalar, got shape ",
                            num_points_tensor.shape().DebugString()));
        const int64 num_points = num_points_tensor.scalar<int64>()();

        const Tensor& inp_neighbors_index = context->input(1);

        const Tensor& inp_neighbors_row_splits = context->input(2);

        const Tensor& inp_neighbors_attributes = context->input(3);

        // check input shapes
        {
            using namespace open3d::ml::op_util;
            Dim num_neighbors("num_neighbors");

            CHECK_SHAPE(context, inp_neighbors_index, num_neighbors);
            CHECK_SHAPE_IGNORE_LAST_DIMS(context, inp_neighbors_attributes,
                                         num_neighbors || 0);
            CHECK_SHAPE(context, inp_neighbors_row_splits, Dim());
        }

        // compute the number of attributes for each neighbor
        int num_attributes;
        if (inp_neighbors_attributes.shape().dim_size(0) == 0) {
            num_attributes = 0;
        } else {
            num_attributes = 1;
            for (int i = 1; i < inp_neighbors_attributes.shape().dims(); ++i)
                num_attributes *= inp_neighbors_attributes.shape().dim_size(i);
        }

        Tensor* neighbors_index = 0;
        TensorShape neighbors_index_shape(inp_neighbors_index.shape());
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, neighbors_index_shape,
                                                &neighbors_index));

        Tensor* neighbors_row_splits = 0;
        TensorShape neighbors_row_splits_shape({num_points + 1});
        OP_REQUIRES_OK(context,
                       context->allocate_output(1, neighbors_row_splits_shape,
                                                &neighbors_row_splits));

        Tensor* neighbors_attributes = 0;
        TensorShape neighbors_attributes_shape(
                inp_neighbors_attributes.shape());
        OP_REQUIRES_OK(context,
                       context->allocate_output(2, neighbors_attributes_shape,
                                                &neighbors_attributes));

        Kernel(context, inp_neighbors_index, inp_neighbors_row_splits,
               inp_neighbors_attributes, num_attributes, *neighbors_index,
               *neighbors_row_splits, *neighbors_attributes);
    }

    // Function with the device specific code
    virtual void Kernel(tensorflow::OpKernelContext* context,
                        const tensorflow::Tensor& inp_neighbors_index,
                        const tensorflow::Tensor& inp_neighbors_row_splits,
                        const tensorflow::Tensor& inp_neighbors_attributes,
                        const int num_attributes,
                        tensorflow::Tensor& neighbors_index,
                        tensorflow::Tensor& neighbors_row_splits,
                        tensorflow::Tensor& neighbors_attributes) = 0;

private:
};

}  // namespace invert_neighbors_list_opkernel
/// @endcond
