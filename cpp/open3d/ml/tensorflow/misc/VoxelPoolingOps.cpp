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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"

using namespace tensorflow;

REGISTER_OP("Open3DVoxelPooling")
        .Attr("TReal: {float, double}")  // type for the point positions
        .Attr("TFeat: {float, double, int32, int64}")  // type for the features
        .Attr("position_fn: {'average', 'nearest_neighbor', 'center'} = "
              "'average'")
        .Attr("feature_fn: {'average', 'nearest_neighbor', 'max'} = 'average'")
        .Attr("debug: bool = false")
        .Input("positions: TReal")
        .Input("features: TFeat")
        .Input("voxel_size: TReal")
        .Output("pooled_positions: TReal")
        .Output("pooled_features: TFeat")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            using namespace ::tensorflow::shape_inference;
            ShapeHandle positions_shape, voxel_size_shape, features_shape,
                    pooled_positions_shape, pooled_features_shape;

            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &positions_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &features_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &voxel_size_shape));

            // we don't know the number of output points
            pooled_positions_shape =
                    c->MakeShape({c->UnknownDim(), c->MakeDim(3)});
            c->set_output(0, pooled_positions_shape);

            DimensionHandle channel_dim = c->UnknownDim();
            if (c->RankKnown(features_shape)) {
                channel_dim = c->Dim(features_shape, -1);
            }
            pooled_features_shape =
                    c->MakeShape({c->UnknownDim(), channel_dim});
            c->set_output(1, pooled_features_shape);

            // check if we have a [N,3] tensor for the positions
            if (c->RankKnown(positions_shape)) {
                DimensionHandle d;
                TF_RETURN_IF_ERROR(
                        c->WithValue(c->Dim(positions_shape, -1), 3, &d));
            }

            return Status::OK();
        })
        .Doc(R"doc(
Spatial pooling for point clouds by combining points that fall into the same voxel bin.

The voxel grid used for pooling is always aligned to the origin (0,0,0) to
simplify building voxel grid hierarchies. The order of the returned voxels is
not defined as can be seen in the following example::

  import open3d.ml.tf as ml3d

  positions = [
      [0.1,0.1,0.1],
      [0.5,0.5,0.5],
      [1.7,1.7,1.7],
      [1.8,1.8,1.8],
      [0.3,2.4,1.4]]

  features = [[1.0,2.0],
              [1.1,2.3],
              [4.2,0.1],
              [1.3,3.4],
              [2.3,1.9]]

  ml3d.ops.voxel_pooling(positions, features, 1.0,
                         position_fn='center', feature_fn='max')

  # or with pytorch
  import torch
  import open3d.ml.torch as ml3d

  positions = torch.Tensor([
      [0.1,0.1,0.1],
      [0.5,0.5,0.5],
      [1.7,1.7,1.7],
      [1.8,1.8,1.8],
      [0.3,2.4,1.4]])

  features = torch.Tensor([
              [1.0,2.0],
              [1.1,2.3],
              [4.2,0.1],
              [1.3,3.4],
              [2.3,1.9]])

  ml3d.ops.voxel_pooling(positions, features, 1.0,
                         position_fn='center', feature_fn='max')

  # returns the voxel centers  [[0.5, 2.5, 1.5],
  #                             [1.5, 1.5, 1.5],
  #                             [0.5, 0.5, 0.5]]
  # and the max pooled features for each voxel [[2.3, 1.9],
  #                                             [4.2, 3.4],
  #                                             [1.1, 2.3]]

position_fn: Defines how the new point positions will be computed.
  The options are
    * "average" computes the center of gravity for the points within one voxel.
    * "nearest_neighbor" selects the point closest to the voxel center.
    * "center" uses the voxel center for the position of the generated point.

feature_fn: Defines how the pooled features will be computed.
  The options are
    * "average" computes the average feature vector.
    * "nearest_neighbor" selects the feature vector of the point closest to the voxel center.
    * "max" uses the maximum feature among all points within the voxel.

debug: If true additional checks for debugging will be enabled.

positions: The point positions with shape [N,3] with N as the number of points.

features: The feature vector with shape [N,channels].

voxel_size: The voxel size.

pooled_positions: The output point positions with shape [M,3] and M <= N.

pooled_features: The output point features with shape [M,channels] and M <= N.

)doc");

REGISTER_OP("Open3DVoxelPoolingGrad")
        .Attr("TReal: {float, double}")  // type for the point positions
        .Attr("TFeat: {float, double, int32, int64}")  // type for the features
        .Attr("position_fn: {'average', 'nearest_neighbor', 'center'} = "
              "'average'")
        .Attr("feature_fn: {'average', 'nearest_neighbor', 'max'} = 'average'")
        .Input("positions: TReal")
        .Input("features: TFeat")
        .Input("voxel_size: TReal")
        .Input("pooled_positions: TReal")
        .Input("pooled_features_gradient: TFeat")
        .Output("features_backprop: TFeat")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            using namespace ::tensorflow::shape_inference;
            ShapeHandle positions_shape, voxel_size_shape, features_shape,
                    pooled_positions_shape, pooled_features_gradient_shape;

            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &positions_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &features_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &voxel_size_shape));
            TF_RETURN_IF_ERROR(
                    c->WithRank(c->input(3), 2, &pooled_positions_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 2,
                                           &pooled_features_gradient_shape));

            c->set_output(0, features_shape);

            DimensionHandle channel_dim = c->UnknownDim();
            if (c->RankKnown(features_shape)) {
                channel_dim = c->Dim(features_shape, -1);
            }
            if (c->RankKnown(pooled_features_gradient_shape)) {
                TF_RETURN_IF_ERROR(c->Merge(
                        channel_dim, c->Dim(pooled_features_gradient_shape, -1),
                        &channel_dim));
            }

            DimensionHandle first_dim = c->UnknownDim();
            if (c->RankKnown(features_shape)) {
                first_dim = c->Dim(features_shape, 0);
            }
            if (c->RankKnown(positions_shape)) {
                TF_RETURN_IF_ERROR(c->Merge(
                        first_dim, c->Dim(positions_shape, 0), &first_dim));
            }

            DimensionHandle first_dim_pooled = c->UnknownDim();
            if (c->RankKnown(pooled_features_gradient_shape)) {
                first_dim_pooled = c->Dim(pooled_features_gradient_shape, 0);
            }
            if (c->RankKnown(pooled_positions_shape)) {
                TF_RETURN_IF_ERROR(c->Merge(first_dim_pooled,
                                            c->Dim(pooled_positions_shape, 0),
                                            &first_dim_pooled));
            }

            // check if we have a [N,3] tensor for the positions
            if (c->RankKnown(positions_shape)) {
                DimensionHandle d;
                TF_RETURN_IF_ERROR(
                        c->WithValue(c->Dim(positions_shape, -1), 3, &d));
            }
            if (c->RankKnown(pooled_positions_shape)) {
                DimensionHandle d;
                TF_RETURN_IF_ERROR(c->WithValue(
                        c->Dim(pooled_positions_shape, -1), 3, &d));
            }

            return Status::OK();
        })
        .Doc(R"doc(
Gradient for features in VoxelPooling. For internal use only.
)doc");
