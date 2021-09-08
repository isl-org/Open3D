// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
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

class RoiPoolOpKernel : public tensorflow::OpKernel {
public:
    explicit RoiPoolOpKernel(tensorflow::OpKernelConstruction* construction)
        : OpKernel(construction) {
        using namespace tensorflow;
        OP_REQUIRES_OK(construction, construction->GetAttr("sampled_pts_num",
                                                           &sampled_pts_num));
    }

    void Compute(tensorflow::OpKernelContext* context) override {
        using namespace tensorflow;

        const Tensor& inp_tensor = context->input(0);
        OP_REQUIRES(
                context,
                inp_tensor.dims() == 3 && inp_tensor.shape().dim_size(2) == 3,
                errors::InvalidArgument("RoiPool expects "
                                        "(batch_size,num_points,3) inp shape"));
        int batch_size = inp_tensor.shape().dim_size(0);
        int pts_num = inp_tensor.shape().dim_size(1);
        auto inp_flat = inp_tensor.flat<float>();
        const float* inp = &(inp_flat(0));

        const Tensor& boxes3d_tensor = context->input(1);
        OP_REQUIRES(context,
                    boxes3d_tensor.dims() == 3 &&
                            boxes3d_tensor.shape().dim_size(2) == 7,
                    errors::InvalidArgument(
                            "RoiPool expects "
                            "(batch_size,num_boxes,7) boxes3d shape"));
        int boxes_num = boxes3d_tensor.shape().dim_size(1);
        auto boxes3d_flat = boxes3d_tensor.flat<float>();
        const float* boxes3d = &(boxes3d_flat(0));

        const Tensor& feats_tensor = context->input(2);
        OP_REQUIRES(context,
                    feats_tensor.dims() == 3 &&
                            feats_tensor.shape().dim_size(1) == pts_num,
                    errors::InvalidArgument(
                            "RoiPool expects "
                            "(batch_size,num_points,feats) feats shape"));
        int feature_in_len = feats_tensor.shape().dim_size(2);
        auto feats_flat = feats_tensor.flat<float>();
        const float* feats = &(feats_flat(0));

        Tensor* out_feats;
        OP_REQUIRES_OK(context,
                       context->allocate_output(
                               0,
                               TensorShape{batch_size, boxes_num,
                                           sampled_pts_num, 3 + feature_in_len},
                               &out_feats));
        auto out_flat0 = out_feats->flat<float>();
        float* out0 = &(out_flat0(0));

        Tensor* out_flags;
        OP_REQUIRES_OK(context, context->allocate_output(
                                        1, TensorShape{batch_size, boxes_num},
                                        &out_flags));
        auto out_flat1 = out_flags->flat<int>();
        int* out1 = &(out_flat1(0));

        Kernel(context, batch_size, pts_num, boxes_num, feature_in_len,
               sampled_pts_num, inp, boxes3d, feats, out0, out1);
    }

    virtual void Kernel(tensorflow::OpKernelContext* context,
                        int batch_size,
                        int pts_num,
                        int boxes_num,
                        int feature_in_len,
                        int sampled_pts_num,
                        const float* xyz,
                        const float* boxes3d,
                        const float* pts_feature,
                        float* pooled_features,
                        int* pooled_empty_flag) = 0;

protected:
    int sampled_pts_num;
};
