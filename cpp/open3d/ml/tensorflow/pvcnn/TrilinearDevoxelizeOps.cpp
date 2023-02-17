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
        .Doc(R"doc(
Trilinear Devoxelize.

This function takes a 3D voxel grid and a list of coordinates and
computes interpolated features corresponding to each point.

Minimal example::
  import open3d.ml.tf as ml3d

  coords = tf.Tensor(
        [[[0.2 0.0 0.0 1.0 1.5]
          [1.0 1.2 0.9 0.0 0.7]
          [0.2 0.6 0.8 0.7 1.1]]], shape=(1, 3, 5), dtype=float32)

  features = tf.Tensor(
  [[[[[0.  0.5]
      [0.6 0.4]]

     [[0.5 0.7]
      [0.6 0.5]]]

    [[[0.4 0.8]
      [0.6 0.3]]

     [[0.4 0.2]
      [0.8 0.6]]]

    [[[0.1 0.2]
      [0.  0.6]]

     [[0.9 0. ]
      [0.2 0.3]]]]], shape=(1, 3, 2, 2, 2), dtype=float32)

  ml3d.ops.trilinear_devoxelize(coords,
                                features,
                                resolution,
                                is_training)

  # returns output tf.Tensor(
  #                array([[[0.564     , 0.508     , 0.436     , 0.64      , 0.5005    ],
  #                        [0.58400005, 0.39200002, 0.396     , 0.26000002, 0.47900003],
  #                        [0.14      , 0.36      , 0.45000002, 0.27      , 0.0975    ]]],
  #                shape=(1, 3, 5), dtype=float32)
  #
  #         indices tf.Tensor([[[ 2,  2,  0,  4,  5],
  #                             [ 3,  3,  1,  5,  6],
  #                             [ 2,  4,  2,  4,  7],
  #                             [ 3,  5,  3,  5,  8],
  #                             [ 6,  2,  0,  4,  9],
  #                             [ 7,  3,  1,  5, 10],
  #                             [ 6,  4,  2,  4, 11],
  #                             [ 7,  5,  3,  5, 12]]],
  #                 shape=(1, 8, 5), dtype=float32)
  #
  #         weights tf.Tensor([[[0.64000005, 0.31999996, 0.02      , 0.3       , 0.135     ],
  #                             [0.16000001, 0.48      , 0.08000002, 0.7       , 0.015     ],
  #                             [0.        , 0.08000001, 0.17999998, 0.        , 0.315     ],
  #                             [0.        , 0.12000003, 0.71999997, 0.        , 0.03500001],
  #                             [0.16000001, 0.        , 0.        , 0.        , 0.135     ],
  #                             [0.04      , 0.        , 0.        , 0.        , 0.015     ],
  #                             [0.        , 0.        , 0.        , 0.        , 0.315     ],
  #                             [0.        , 0.        , 0.        , 0.        , 0.03500001]]],
  #                 shape=(1, 8, 5), dtype=float32)

coords: List of 3D coordinates for which features to be interpolated.
  The shape of this tensor is [B, 3, N]. The range of coordinates is
  [0, resolution-1]. If all of the adjacent position of any coordinate are out
  of range, then the interpolated features will be 0. Voxel centers are at integral
  values of voxel grid.

features: A voxel grid of shape [B, C, R, R, R]. Here R is resolution.

resolution: Integer attribute defining resolution of the voxel grid.

is_training: Boolean variable for training phase.

outputs: Features for each point. The shape of this tensor is [B, C, N].

indices: Indices which are used to interpolate features. Shape is [B, 8, N].

weights: Weights for each index used to interpolate features. Shape is [B, 8, N].

)doc");

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
        .Doc(R"doc(
Gradient function for Trilinear Devoxelize op.

This function takes feature gradients and indices, weights returned from
the op and computes gradient for voxelgrid.

grad_y: Gradients for the interpolated features. Shape is [B, C, N].

indices: Indices which are used to interpolate features. Shape is [B, 8, N].

weights: Weights for each index used to interpolate features. Shape is [B, 8, N].

resolution: Integer attribute defining resolution of the grid.

grad_x: Output gradients for the voxel grid. Shape is [B, C, R, R, R]

)doc");
