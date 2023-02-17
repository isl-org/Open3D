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
            return Status::OK();
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
            return Status::OK();
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
            return Status::OK();
        })
        .Doc(R"doc( TODO )doc");
