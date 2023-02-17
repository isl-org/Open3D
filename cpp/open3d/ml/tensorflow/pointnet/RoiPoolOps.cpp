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
            return Status::OK();
        })
        .Doc(R"doc( TODO )doc");
