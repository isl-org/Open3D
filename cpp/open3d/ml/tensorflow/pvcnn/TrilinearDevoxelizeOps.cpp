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

#include "open3d/ml/tensorflow/TensorFlowHelper.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("Open3DTrilinearDevoxelize")
        .Attr("resolution: int")
        .Attr("is_training: bool")
        .Input("coords: float32")
        .Input("features: float32")
        .Output("outputs: float32")
        .Output("indices: int32")
        .Output("weights: float32")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            using namespace ::tensorflow::shape_inference;
            using namespace open3d::ml::op_util;
            ShapeHandle coords;    // (batch_size, 3, N)
            ShapeHandle features;  // (batch_size, C, R, R, R)

            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &coords));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 5, &features));

            Dim batch_size("batch_size");
            Dim num_points("num_points");
            Dim resolution("resolution");
            Dim feat_dim("feat_dim");
            CHECK_SHAPE_HANDLE(c, coords, batch_size, 3, num_points);
            CHECK_SHAPE_HANDLE(c, features, batch_size, feat_dim, resolution,
                               resolution, resolution);

            ShapeHandle out1, out2;
            out1 = c->MakeShape(
                    {c->MakeDim(batch_size.value()),
                     c->MakeDim(feat_dim.value()),
                     c->MakeDim(num_points.value())});  // (batch_size, C, N)
            out2 = c->MakeShape(
                    {c->MakeDim(batch_size.value()), 8,
                     c->MakeDim(num_points.value())});  // (batch_size, 8, N)

            c->set_output(0, out1);
            c->set_output(1, out2);
            c->set_output(2, out2);

            return Status::OK();
        })
        .Doc(R"doc( TODO )doc");

REGISTER_OP("Open3DTrilinearDevoxelizeGrad")
        .Input("grad_y: float32")
        .Input("indices: int32")
        .Input("weights: float32")
        .Attr("resolution: int")
        .Output("grad_x: float32")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            using namespace ::tensorflow::shape_inference;
            using namespace open3d::ml::op_util;
            ShapeHandle grad_y;   // (batch_size, C, N)
            ShapeHandle indices;  // (batch_size, 8, N)

            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &grad_y));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &indices));

            Dim batch_size("batch_size");
            Dim feat_dim("feat_dim");
            Dim num_points("num_points");
            CHECK_SHAPE_HANDLE(c, grad_y, batch_size, feat_dim, num_points);
            CHECK_SHAPE_HANDLE(c, indices, batch_size, 8, num_points);

            ShapeHandle out;
            int r_;
            TF_RETURN_IF_ERROR(c->GetAttr("resolution", &r_));
            out = c->MakeShape({c->MakeDim(batch_size.value()),
                                c->MakeDim(feat_dim.value()), r_, r_,
                                r_});  // (batch_size, C, R, R, R)

            c->set_output(0, out);

            return Status::OK();
        })
        .Doc(R"doc( TODO )doc");
