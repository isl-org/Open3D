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

REGISTER_OP("Open3DThreeNN")
        .Input("query_pts: float32")
        .Input("data_pts: float32")
        .Output("out_dist2: float32")
        .Output("out_idx: int32")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle
                    dims1;  // batch_size * nsample * 3
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &dims1));
            ::tensorflow::shape_inference::ShapeHandle output =
                    c->MakeShape({c->Dim(dims1, 0), c->Dim(dims1, 1), 3});
            c->set_output(0, output);
            c->set_output(1, output);
            return Status();
        })
        .Doc(R"doc( TODO )doc");

REGISTER_OP("Open3DThreeInterpolate")
        .Input("points: float32")
        .Input("idx: int32")
        .Input("weights: float32")
        .Output("out: float32")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle
                    dims1;  // batch_size * nsample * 3
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &dims1));
            ::tensorflow::shape_inference::ShapeHandle
                    dims2;  // batch_size * nsample * 3
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &dims2));

            ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape(
                    {c->Dim(dims1, 0), c->Dim(dims1, 1), c->Dim(dims2, 1)});
            c->set_output(0, output);
            return Status();
        })
        .Doc(R"doc( TODO )doc");

REGISTER_OP("Open3DThreeInterpolateGrad")
        .Input("grad_out: float32")
        .Input("idx: int32")
        .Input("weights: float32")
        .Attr("M: int")
        .Output("out: float32")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle
                    dims1;  // batch_size * nsample * 3
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &dims1));
            int M;
            TF_RETURN_IF_ERROR(c->GetAttr("M", &M));
            ::tensorflow::shape_inference::ShapeHandle output =
                    c->MakeShape({c->Dim(dims1, 0), c->Dim(dims1, 1), M});
            c->set_output(0, output);
            c->set_output(1, output);
            return Status();
        })
        .Doc(R"doc( TODO )doc");
