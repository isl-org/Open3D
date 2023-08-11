// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
/* Furthest point sampling
 * Original author: Haoqiang Fan
 * Modified by Charles R. Qi
 * All Rights Reserved. 2017.
 */

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("Open3DBallQuery")
        .Input("xyz: float32")
        .Input("center: float32")
        .Attr("radius: float")
        .Attr("nsample: int")
        .Output("out: int32")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle
                    dims1;  // batch_size * nsample * 3
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &dims1));
            int nsample;
            TF_RETURN_IF_ERROR(c->GetAttr("nsample", &nsample));
            ::tensorflow::shape_inference::ShapeHandle output =
                    c->MakeShape({c->Dim(dims1, 0), c->Dim(dims1, 1), nsample});
            c->set_output(0, output);
            return Status();
        })
        .Doc(R"doc( TODO )doc");
