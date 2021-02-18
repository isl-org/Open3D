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

class FurthestPointSamplingOpKernel : public tensorflow::OpKernel {
public:
    explicit FurthestPointSamplingOpKernel(
            tensorflow::OpKernelConstruction* construction)
        : OpKernel(construction) {
        using namespace tensorflow;

        OP_REQUIRES_OK(construction,
                       construction->GetAttr("sample_size", &sample_size));
        OP_REQUIRES(construction, sample_size > 0,
                    errors::InvalidArgument(
                            "FurthestPointSampling expects positive npoint"));
    }

    void Compute(tensorflow::OpKernelContext* context) override {
        using namespace tensorflow;

        const Tensor& inp_tensor = context->input(0);
        OP_REQUIRES(
                context,
                inp_tensor.dims() == 3 && inp_tensor.shape().dim_size(2) == 3,
                errors::InvalidArgument("FurthestPointSampling expects "
                                        "(batch_size,num_points,3) inp shape"));
        int batch_size = inp_tensor.shape().dim_size(0);
        int pts_size = inp_tensor.shape().dim_size(1);
        auto inp_flat = inp_tensor.flat<float>();
        const float* inp = &(inp_flat(0));

        Tensor* out_tensor;
        OP_REQUIRES_OK(context, context->allocate_output(
                                        0, TensorShape{batch_size, sample_size},
                                        &out_tensor));
        auto out_flat = out_tensor->flat<int>();
        int* out = &(out_flat(0));

        Tensor temp_tensor;
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DataTypeToEnum<float>::value,
                                              TensorShape{batch_size, pts_size},
                                              &temp_tensor));
        auto temp_flat = temp_tensor.flat<float>();
        float* temp = &(temp_flat(0));
        // fill with big value
        cudaMemset(temp, 80, batch_size * pts_size * sizeof(float));

        Kernel(context, batch_size, pts_size, sample_size, inp, temp, out);
    }

    virtual void Kernel(tensorflow::OpKernelContext* context,
                        int b,
                        int n,
                        int m,
                        const float* dataset,
                        float* temp,
                        int* idxs) = 0;

protected:
    int sample_size;
};

class GatherPointsOpKernel : public tensorflow::OpKernel {
public:
    explicit GatherPointsOpKernel(
            tensorflow::OpKernelConstruction* construction)
        : OpKernel(construction) {}

    void Compute(tensorflow::OpKernelContext* context) override {
        using namespace tensorflow;

        const Tensor& inp_tensor = context->input(0);
        OP_REQUIRES(
                context, inp_tensor.dims() == 3,
                errors::InvalidArgument("GatherPoints expects "
                                        "(batch_size,num_points,3) inp shape"));
        int batch_size = inp_tensor.shape().dim_size(0);
        int group_size = inp_tensor.shape().dim_size(1);
        int feature_size = inp_tensor.shape().dim_size(2);
        auto inp_flat = inp_tensor.flat<float>();
        const float* inp = &(inp_flat(0));

        const Tensor& idx_tensor = context->input(1);
        OP_REQUIRES(
                context,
                idx_tensor.dims() == 2 &&
                        idx_tensor.shape().dim_size(0) == batch_size,
                errors::InvalidArgument("GatherPoints expects "
                                        "(batch_size,num_result) idx shape"));
        int idx_size = idx_tensor.shape().dim_size(1);
        auto idx_flat = idx_tensor.flat<int>();
        const int* idx = &(idx_flat(0));

        Tensor* out_tensor = NULL;
        OP_REQUIRES_OK(context,
                       context->allocate_output(
                               0, TensorShape{batch_size, group_size, idx_size},
                               &out_tensor));
        auto out_flat = out_tensor->flat<float>();
        float* out = &(out_flat(0));

        Kernel(context, batch_size, group_size, feature_size, idx_size, inp,
               idx, out);
    }

    virtual void Kernel(tensorflow::OpKernelContext* context,
                        int b,
                        int c,
                        int n,
                        int npoints,
                        const float* points,
                        const int* idx,
                        float* out) = 0;
};

class GatherPointsGradOpKernel : public tensorflow::OpKernel {
public:
    explicit GatherPointsGradOpKernel(
            tensorflow::OpKernelConstruction* construction)
        : OpKernel(construction) {
        OP_REQUIRES_OK(construction, construction->GetAttr("N", &N));
        OP_REQUIRES_OK(construction, construction->GetAttr("C", &C));
    }

    void Compute(tensorflow::OpKernelContext* context) override {
        using namespace tensorflow;

        const Tensor& idx_tensor = context->input(0);
        OP_REQUIRES(
                context, idx_tensor.dims() == 2,
                errors::InvalidArgument("GatherPointsGradGpuOp expects "
                                        "(batch_size,num_result) idx shape"));
        int batch_size = idx_tensor.shape().dim_size(0);
        int idx_size = idx_tensor.shape().dim_size(1);
        auto idx_flat = idx_tensor.flat<int>();
        const int* idx = &(idx_flat(0));

        const Tensor& out_g_tensor = context->input(2);
        OP_REQUIRES(context,
                    out_g_tensor.dims() == 3 &&
                            out_g_tensor.shape().dim_size(0) == batch_size,
                    errors::InvalidArgument(
                            "GatherPointsGradGpuOp expects "
                            "(batch_size,num_result,3) out_g shape"));
        auto out_g_flat = out_g_tensor.flat<float>();
        const float* out_g = &(out_g_flat(0));

        Tensor* inp_g_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(
                                        0, TensorShape{batch_size, C, N},
                                        &inp_g_tensor));
        auto inp_g_flat = inp_g_tensor->flat<float>();
        float* inp_g = &(inp_g_flat(0));
        cudaMemset(inp_g, 0, batch_size * C * N * sizeof(float));

        Kernel(context, batch_size, C, N, idx_size, out_g, idx, inp_g);
    }

    virtual void Kernel(tensorflow::OpKernelContext* context,
                        int b,
                        int c,
                        int n,
                        int npoints,
                        const float* grad_out,
                        const int* idx,
                        float* grad_points) = 0;

protected:
    int C, N;
};