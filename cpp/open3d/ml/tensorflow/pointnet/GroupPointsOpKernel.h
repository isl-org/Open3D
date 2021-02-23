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

#include "../TensorFlowHelper.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"

class GroupPointsOpKernel : public tensorflow::OpKernel {
public:
    explicit GroupPointsOpKernel(tensorflow::OpKernelConstruction* construction)
        : OpKernel(construction) {}

    void Compute(tensorflow::OpKernelContext* context) override {
        using namespace tensorflow;

        const Tensor& inp_tensor = context->input(0);
        OP_REQUIRES(context, inp_tensor.dims() == 3,
                    errors::InvalidArgument(
                            "GroupPoints expects "
                            "(batch_size,num_points,num_features) inp shape"));
        int batch_size = inp_tensor.shape().dim_size(0);
        int C = inp_tensor.shape().dim_size(1);
        int N = inp_tensor.shape().dim_size(2);
        auto inp_flat = inp_tensor.flat<float>();
        const float* inp = &(inp_flat(0));

        const Tensor& idx_tensor = context->input(1);
        OP_REQUIRES(context, idx_tensor.dims() == 3,
                    errors::InvalidArgument(
                            "GroupPoints expects "
                            "(batch_size,group_size,sample_size) idx shape"));
        int feature_size = idx_tensor.shape().dim_size(1);
        int sample_size = idx_tensor.shape().dim_size(2);
        auto idx_flat = idx_tensor.flat<int>();
        const int* idx = &(idx_flat(0));

        Tensor* out_tensor;
        OP_REQUIRES_OK(context, context->allocate_output(
                                        0,
                                        TensorShape{batch_size, C, feature_size,
                                                    sample_size},
                                        &out_tensor));
        auto out_flat = out_tensor->flat<float>();
        float* out = &(out_flat(0));

        Kernel(context, batch_size, C, N, feature_size, sample_size, inp, idx,
               out);
    }

    virtual void Kernel(tensorflow::OpKernelContext* context,
                        int b,
                        int c,
                        int n,
                        int npoints,
                        int nsample,
                        const float* points,
                        const int* idx,
                        float* out) = 0;
};

class GroupPointsGradOpKernel : public tensorflow::OpKernel {
public:
    explicit GroupPointsGradOpKernel(
            tensorflow::OpKernelConstruction* construction)
        : OpKernel(construction) {
        using namespace tensorflow;

        OP_REQUIRES_OK(construction, construction->GetAttr("N", &N));
    }

    void Compute(tensorflow::OpKernelContext* context) override {
        using namespace tensorflow;

        const Tensor& inp_tensor = context->input(0);
        int batch_size = inp_tensor.shape().dim_size(0);
        int C = inp_tensor.shape().dim_size(1);
        int feature_size = inp_tensor.shape().dim_size(2);
        int sample_size = inp_tensor.shape().dim_size(3);
        auto inp_flat = inp_tensor.flat<float>();
        const float* inp = &(inp_flat(0));

        const Tensor& idx_tensor = context->input(1);
        OP_REQUIRES(context, idx_tensor.dims() == 3,
                    errors::InvalidArgument(
                            "GroupPointsGrad expects "
                            "(batch_size,group_size,sample_size) idx shape"));
        auto idx_flat = idx_tensor.flat<int>();
        const int* idx = &(idx_flat(0));

        Tensor* out_tensor;
        OP_REQUIRES_OK(context,
                       context->allocate_output(
                               0, TensorShape{batch_size, C, N}, &out_tensor));
        auto out_flat = out_tensor->flat<float>();
        float* out = &(out_flat(0));

        Kernel(context, batch_size, C, N, feature_size, sample_size, inp, idx,
               out);
    }

    virtual void Kernel(tensorflow::OpKernelContext* context,
                        int b,
                        int c,
                        int n,
                        int npoints,
                        int nsample,
                        const float* grad_out,
                        const int* idx,
                        float* grad_points) = 0;

protected:
    int N;
};
