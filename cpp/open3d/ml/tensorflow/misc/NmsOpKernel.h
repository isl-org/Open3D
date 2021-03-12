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

//#include "open3d/ml/impl/misc/VoxelPooling.h"
#include "open3d/ml/tensorflow/TensorFlowHelper.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"

/// @cond
// namespace for code that is common for all kernels
namespace nms_opkernel {

class OutputAllocator {
public:
    OutputAllocator(tensorflow::OpKernelContext* context) : context(context) {}

    void AllocKeepIndices(int64_t** ptr, int64_t num) {
        using namespace tensorflow;
        *ptr = nullptr;
        Tensor* tensor = 0;
        TensorShape shape({num});
        OP_REQUIRES_OK(context, context->allocate_output(0, shape, &tensor));
        auto flat_tensor = tensor->flat<int64>();
        *ptr = (int64_t*)flat_tensor.data();
    }

private:
    tensorflow::OpKernelContext* context;
};

// Base class with common code for the OpKernel implementations
class NmsOpKernel : public tensorflow::OpKernel {
public:
    explicit NmsOpKernel(tensorflow::OpKernelConstruction* construction)
        : OpKernel(construction) {
        OP_REQUIRES_OK(construction,
                       construction->GetAttr("nms_overlap_thresh",
                                             &nms_overlap_thresh));
    }

    void Compute(tensorflow::OpKernelContext* context) override {
        using namespace tensorflow;
        const Tensor& boxes = context->input(0);
        const Tensor& scores = context->input(1);

        {
            using namespace open3d::ml::op_util;
            Dim num_points("num_points");
            Dim five(5, "five");
            CHECK_SHAPE(context, boxes, num_points, five);
            CHECK_SHAPE(context, scores, num_points);
        }

        Kernel(context, boxes, scores);
    }

    // Function with the device specific code
    virtual void Kernel(tensorflow::OpKernelContext* context,
                        const tensorflow::Tensor& boxes,
                        const tensorflow::Tensor& scores) = 0;

protected:
    float nms_overlap_thresh;
};

}  // namespace nms_opkernel
/// @endcond
