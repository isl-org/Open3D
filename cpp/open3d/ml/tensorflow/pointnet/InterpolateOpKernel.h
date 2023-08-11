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

class ThreeNNOpKernel : public tensorflow::OpKernel {
public:
    explicit ThreeNNOpKernel(tensorflow::OpKernelConstruction* construction)
        : OpKernel(construction) {}

    void Compute(tensorflow::OpKernelContext* context) override {
        using namespace tensorflow;

        const Tensor& inp_tensor = context->input(0);
        OP_REQUIRES(
                context,
                inp_tensor.dims() == 3 && inp_tensor.shape().dim_size(2) == 3,
                errors::InvalidArgument("ThreeNN expects "
                                        "(batch_size,num_points,3) inp shape"));
        int batch_size = inp_tensor.shape().dim_size(0);
        int pts_num_out = inp_tensor.shape().dim_size(1);
        auto inp_flat = inp_tensor.flat<float>();
        const float* inp = &(inp_flat(0));

        const Tensor& data_tensor = context->input(1);
        OP_REQUIRES(
                context,
                data_tensor.dims() == 3 && data_tensor.shape().dim_size(2) == 3,
                errors::InvalidArgument(
                        "ThreeNN expects "
                        "(batch_size,num_points,3) data shape"));
        int pts_num_in = data_tensor.shape().dim_size(1);
        auto data_flat = data_tensor.flat<float>();
        const float* data = &(data_flat(0));

        Tensor* out_dist;
        OP_REQUIRES_OK(
                context,
                context->allocate_output(
                        0, TensorShape{batch_size, pts_num_out, 3}, &out_dist));
        auto out_flat0 = out_dist->flat<float>();
        float* out0 = &(out_flat0(0));

        Tensor* out_idx;
        OP_REQUIRES_OK(
                context,
                context->allocate_output(
                        1, TensorShape{batch_size, pts_num_out, 3}, &out_idx));
        auto out_flat1 = out_idx->flat<int>();
        int* out1 = &(out_flat1(0));

        Kernel(context, batch_size, pts_num_out, pts_num_in, inp, data, out0,
               out1);
    }

    virtual void Kernel(tensorflow::OpKernelContext* context,
                        int b,
                        int n,
                        int m,
                        const float* unknown,
                        const float* known,
                        float* dist2,
                        int* idx) = 0;
};

class ThreeInterpolateOpKernel : public tensorflow::OpKernel {
public:
    explicit ThreeInterpolateOpKernel(
            tensorflow::OpKernelConstruction* construction)
        : OpKernel(construction) {}

    void Compute(tensorflow::OpKernelContext* context) override {
        using namespace tensorflow;

        const Tensor& inp_tensor = context->input(0);
        OP_REQUIRES(
                context, inp_tensor.dims() == 3,
                errors::InvalidArgument("ThreeInterpolate expects "
                                        "(batch_size,num_points,3) inp shape"));
        int batch_size = inp_tensor.shape().dim_size(0);
        int C = inp_tensor.shape().dim_size(1);
        int M = inp_tensor.shape().dim_size(2);
        auto inp_flat = inp_tensor.flat<float>();
        const float* inp = &(inp_flat(0));

        const Tensor& idx_tensor = context->input(1);
        OP_REQUIRES(
                context, idx_tensor.dims() == 3,
                errors::InvalidArgument("ThreeInterpolate expects "
                                        "(batch_size,num_points,3) idx shape"));
        int N = idx_tensor.shape().dim_size(1);
        auto idx_flat = idx_tensor.flat<int>();
        const int* idx = &(idx_flat(0));

        const Tensor& weights_tensor = context->input(2);
        OP_REQUIRES(context, weights_tensor.dims() == 3,
                    errors::InvalidArgument(
                            "ThreeInterpolate expects "
                            "(batch_size,num_points,3) weights shape"));
        auto weights_flat = weights_tensor.flat<float>();
        const float* weights = &(weights_flat(0));

        Tensor* out_tensor;
        OP_REQUIRES_OK(context,
                       context->allocate_output(
                               0, TensorShape{batch_size, C, N}, &out_tensor));
        auto out_flat = out_tensor->flat<float>();
        float* out = &(out_flat(0));

        Kernel(context, batch_size, C, M, N, inp, idx, weights, out);
    }

    virtual void Kernel(tensorflow::OpKernelContext* context,
                        int b,
                        int c,
                        int m,
                        int n,
                        const float* points,
                        const int* idx,
                        const float* weight,
                        float* out) = 0;
};

class ThreeInterpolateGradOpKernel : public tensorflow::OpKernel {
public:
    explicit ThreeInterpolateGradOpKernel(
            tensorflow::OpKernelConstruction* construction)
        : OpKernel(construction) {
        OP_REQUIRES_OK(construction, construction->GetAttr("M", &M));
    }

    void Compute(tensorflow::OpKernelContext* context) override {
        using namespace tensorflow;

        const Tensor& inp_tensor = context->input(0);
        OP_REQUIRES(
                context, inp_tensor.dims() == 3,
                errors::InvalidArgument("ThreeInterpolateGrad expects "
                                        "(batch_size,num_points,3) inp shape"));
        int batch_size = inp_tensor.shape().dim_size(0);
        int C = inp_tensor.shape().dim_size(1);
        int N = inp_tensor.shape().dim_size(2);
        auto inp_flat = inp_tensor.flat<float>();
        const float* inp = &(inp_flat(0));

        const Tensor& idx_tensor = context->input(1);
        OP_REQUIRES(
                context, idx_tensor.dims() == 3,
                errors::InvalidArgument("ThreeInterpolateGrad expects "
                                        "(batch_size,num_points,3) idx shape"));
        auto idx_flat = idx_tensor.flat<int>();
        const int* idx = &(idx_flat(0));

        const Tensor& weights_tensor = context->input(2);
        OP_REQUIRES(context, weights_tensor.dims() == 3,
                    errors::InvalidArgument(
                            "ThreeInterpolateGrad expects "
                            "(batch_size,num_points,3) weights shape"));
        auto weights_flat = weights_tensor.flat<float>();
        const float* weights = &(weights_flat(0));

        Tensor* out_tensor;
        OP_REQUIRES_OK(context,
                       context->allocate_output(
                               0, TensorShape{batch_size, C, M}, &out_tensor));
        auto out_flat = out_tensor->flat<float>();
        float* out = &(out_flat(0));

        Kernel(context, batch_size, C, N, M, inp, idx, weights, out);
    }

    virtual void Kernel(tensorflow::OpKernelContext* context,
                        int b,
                        int c,
                        int n,
                        int m,
                        const float* grad_out,
                        const int* idx,
                        const float* weight,
                        float* grad_points) = 0;

protected:
    int M;
};
