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

REGISTER_OP("Open3DRoiPool")
        .Input("xyz: float32")
        .Input("boxes3d: float32")
        .Input("pts_feature: float32")
        .Attr("sampled_pts_num: int")
        .Output("feats: float32")
        .Output("flags: int32")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle
                    dims1;  // batch_size * nboxes * 7
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &dims1));
            ::tensorflow::shape_inference::ShapeHandle
                    dims2;  // batch_size * nsample * feats
            TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 3, &dims2));

            int sampled_pts_num;
            TF_RETURN_IF_ERROR(c->GetAttr("sampled_pts_num", &sampled_pts_num));

            ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape(
                    {c->Dim(dims1, 0), c->Dim(dims1, 1), sampled_pts_num, -1});
            c->set_output(0, output1);

            ::tensorflow::shape_inference::ShapeHandle output2 =
                    c->MakeShape({c->Dim(dims1, 0), c->Dim(dims1, 1)});
            c->set_output(1, output2);
            return Status();
        })
        .Doc(R"doc( TODO )doc");
