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
