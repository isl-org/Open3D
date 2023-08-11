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

class BallQueryOpKernel : public tensorflow::OpKernel {
public:
    explicit BallQueryOpKernel(tensorflow::OpKernelConstruction* construction)
        : OpKernel(construction) {
        using namespace tensorflow;

        OP_REQUIRES_OK(construction,
                       construction->GetAttr("nsample", &nsample));
        OP_REQUIRES_OK(construction, construction->GetAttr("radius", &radius));
        OP_REQUIRES(
                construction, nsample > 0,
                errors::InvalidArgument("BallQuery expects positive nsample"));
    }

    void Compute(tensorflow::OpKernelContext* context) override {
        using namespace tensorflow;

        const Tensor& inp_tensor = context->input(0);
        OP_REQUIRES(
                context,
                inp_tensor.dims() == 3 && inp_tensor.shape().dim_size(2) == 3,
                errors::InvalidArgument("BallQuery expects "
                                        "(batch_size,num_points,3) inp shape"));
        int batch_size = inp_tensor.shape().dim_size(0);
        int pts_size = inp_tensor.shape().dim_size(1);
        auto inp_flat = inp_tensor.flat<float>();
        const float* inp = &(inp_flat(0));

        const Tensor& center_tensor = context->input(1);
        OP_REQUIRES(context,
                    center_tensor.dims() == 3 &&
                            center_tensor.shape().dim_size(2) == 3,
                    errors::InvalidArgument(
                            "BallQuery expects "
                            "(batch_size,num_points,3) center shape"));
        int ball_size = center_tensor.shape().dim_size(1);
        auto center_flat = center_tensor.flat<float>();
        const float* center = &(center_flat(0));

        Tensor* out_tensor;
        OP_REQUIRES_OK(context,
                       context->allocate_output(
                               0, TensorShape{batch_size, ball_size, nsample},
                               &out_tensor));
        auto out_flat = out_tensor->flat<int>();
        int* out = &(out_flat(0));

        Kernel(context, batch_size, pts_size, ball_size, radius, nsample,
               center, inp, out);
    }

    virtual void Kernel(tensorflow::OpKernelContext* context,
                        int b,
                        int n,
                        int m,
                        float radius,
                        int nsample,
                        const float* new_xyz,
                        const float* xyz,
                        int* idx) = 0;

protected:
    int nsample;
    float radius;
};
