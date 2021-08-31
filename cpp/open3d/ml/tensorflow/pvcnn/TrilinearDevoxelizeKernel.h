// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

class TrilinearDevoxelizeOpKernel : public tensorflow::OpKernel {
public:
    explicit TrilinearDevoxelizeOpKernel(
            tensorflow::OpKernelConstruction* context)
        : tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext* context) override {
        using namespace tensorflow;
        const Tensor& coords = context->input(0);
        OP_REQUIRES(
                context, coords.dim() == 3 && coords.shape().dim_size(1) == 3,
                errors::InvalidArgument("TrilinearDevoxelize expects "
                                        "(batch_size, 3, N) coordinate shape"));
        const Tensor& feat = context->input(1);
        OP_REQUIRES(context, feat.dim() == 5,
                    errors::InvalidArgument("TrilinearDevoxelize expects "
                                            "5 dimensions for features"));

        int batch_size = coords.shape().dim_size(0);
        int num_points = coords.shape().dim_size(2);
        int feat_dim = feat.shape().dim_size(1);
        int r = context->GetAttr("r", &r);
        bool is_training = context->GetAttr("is_training", &is_training);

        auto coords_flat = coords.flat<float32_t>();
        auto feat_flat = feat.flat<float32_t>();

        const float* inp_coords = &(coords_flat(0));
        const float* inp_feat = &(feat_flat(0));

        Tensor* out_tensor_0;
        OP_REQUIRES_OK(context,
                       context->allocate_output(
                               0, TensorShape{batch_size, feat_dim, num_points},
                               &out_tensor_0));
        Tensor* out_tensor_1;
        OP_REQUIRES_OK(context,
                       context->allocate_output(
                               1, TensorShape{batch_size, 8, num_points},
                               &out_tensor_1));
        Tensor* out_tensor_2;
        OP_REQUIRES_OK(context,
                       context->allocate_output(
                               2, TensorShape{batch_size, 8, num_points},
                               &out_tensor_2));
        auto flat_0 = out_tensor_0->flat<float32_t>();
        auto flat_1 = out_tensor_1->flat<int32_t>();
        auto flat_2 = out_tensor_2->flat<float32_t>();

        float32_t* out_0 = &(flat_0(0));
        float32_t* out_1 = &(flat_1(0));
        float32_t* out_2 = &(flat_2(0));

        if (is_training)
            Kernel(context, batch_size, feat_dim, num_points, r, r * r,
                   r * r * r, true, inp_coords, inp_feat, out_1, out_2, out_0);
        else
            Kernel(context, batch_size, feat_dim, num_points, r, r * r,
                   r * r * r, false, inp_coords, inp_feat, out_1, out_2, out_0);
    }

    virtual void Kernel(tensorflow::OpKernelContext* context,
                        int b,
                        int c,
                        int n,
                        int r,
                        int r2,
                        int r3,
                        bool training,
                        const float* coords,
                        const float* feat,
                        int* inds,
                        float* wgts,
                        float* outs) = 0;
};

class TrilinearDevoxelizeGradOpKernel : public tensorflow::OpKernel {
public:
    explicit TrilinearDevoxelizeGradOpKernel(
            tensorflow::OpKernelConstruction* context)
        : tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext* context) override {
        using namespace tensorflow;
        const Tensor& grad_y = context->input(0);
        OP_REQUIRES(
                context, grad_y.dim() == 3,
                errors::InvalidArgument("TrilinearDevoxelizeGrad expects "
                                        "(batch_size, C, N) gradient shape"));
        const Tensor& inds = context->input(1);
        OP_REQUIRES(
                context, inds.dim() == 3 && inds.shape().dim_size(1) == 8,
                errors::InvalidArgument("TrilinearDevoxelizeGrad expects "
                                        "(batch_size, 8, N) indices shape"));
        const Tensor& wgts = context->input(2);
        OP_REQUIRES(
                context, wgts.dim() == 3 && wgts.shape().dim_size(1) == 8,
                errors::InvalidArgument("TrilinearDevoxelizeGrad expects "
                                        "(batch_size, 8, N) weights shape"));

        int batch_size = grad_y.shape().dim_size(0);
        int num_points = grad_y.shape().dim_size(2);
        int feat_dim = grad_y.shape().dim_size(1);
        int r = context->GetAttr("r", &r);

        auto grad_y_flat = grad_y.flat<float32_t>();
        auto inds_flat = inds.flat<int32_t>();
        auto wgts_flat = wgts.flat<float32_t>();

        const float* inp_grad_y = &(grad_y_flat(0));
        const float* inp_inds = &(inds_flat(0));
        const float* inp_wgts = &(wgts_flat(0));

        Tensor* out_tensor;
        OP_REQUIRES_OK(context,
                       context->allocate_output(
                               0, TensorShape{batch_size, feat_dim, r, r, r},
                               &out_tenosr));
        auto flat_tensor = out_tensor->flat<float32_t>();

        float32_t* out = &(flat_tensor(0));

        Kernel(context, batch_size, feat_dim, num_points, r * r * r, inp_inds,
               inp_wgts, inp_grad_y, out);
    }

    virtual void Kernel(tensorflow::OpKernelContext* context,
                        int b,
                        int c,
                        int n,
                        int r3,
                        const int* inds,
                        const float* wgts,
                        const float* grad_y,
                        float* grad_x) = 0;
};
