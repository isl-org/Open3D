// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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
#pragma once

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"

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
        OP_REQUIRES(context, inp_neighbors_index.shape().dims() == 1,
                    errors::InvalidArgument(
                            "inp_neighbors_index must be a rank 1 tensor"));

        const Tensor& inp_neighbors_prefix_sum = context->input(2);
        OP_REQUIRES(
                context, inp_neighbors_prefix_sum.shape().dims() == 1,
                errors::InvalidArgument(
                        "inp_neighbors_prefix_sum must be a rank 1 tensor"));

        const Tensor& inp_neighbors_attributes = context->input(3);
        OP_REQUIRES(context, inp_neighbors_attributes.shape().dims() >= 1,
                    errors::InvalidArgument("inp_neighbors_attributes must be "
                                            "at least a rank 1 tensor"));
        OP_REQUIRES(context,
                    inp_neighbors_attributes.shape().dim_size(0) ==
                                    inp_neighbors_index.shape().dim_size(0) ||
                            inp_neighbors_attributes.shape().dim_size(0) == 0,
                    errors::InvalidArgument(
                            "first dim of inp_neighbors_attributes does not "
                            "match the first dim of inp_neighbors_index"));

        // compute the number of attributes for each neighbor
        int num_attributes;
        if ( inp_neighbors_attributes.shape().dim_size(0) == 0 ) {
          num_attributes = 0;
        }
        else {
          num_attributes = 1; 
          for (int i = 1; i < inp_neighbors_attributes.shape().dims(); ++i)
              num_attributes *= inp_neighbors_attributes.shape().dim_size(i);
        }

        Tensor* neighbors_index = 0;
        TensorShape neighbors_index_shape(inp_neighbors_index.shape());
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, neighbors_index_shape,
                                                &neighbors_index));

        Tensor* neighbors_prefix_sum = 0;
        TensorShape neighbors_prefix_sum_shape({num_points});
        OP_REQUIRES_OK(context,
                       context->allocate_output(1, neighbors_prefix_sum_shape,
                                                &neighbors_prefix_sum));
        Tensor* neighbors_attributes = 0;
        TensorShape neighbors_attributes_shape(
                inp_neighbors_attributes.shape());
        OP_REQUIRES_OK(context,
                       context->allocate_output(2, neighbors_attributes_shape,
                                                &neighbors_attributes));

        Kernel(context, inp_neighbors_index, inp_neighbors_prefix_sum,
               inp_neighbors_attributes, num_attributes, *neighbors_index,
               *neighbors_prefix_sum, *neighbors_attributes);
    }

    // Function with the device specific code
    virtual void Kernel(tensorflow::OpKernelContext* context,
                        const tensorflow::Tensor& inp_neighbors_index,
                        const tensorflow::Tensor& inp_neighbors_prefix_sum,
                        const tensorflow::Tensor& inp_neighbors_attributes,
                        const int num_attributes,
                        tensorflow::Tensor& neighbors_index,
                        tensorflow::Tensor& neighbors_prefix_sum,
                        tensorflow::Tensor& neighbors_attributes) = 0;

private:
};

}  // namespace invert_neighbors_list_opkernel
