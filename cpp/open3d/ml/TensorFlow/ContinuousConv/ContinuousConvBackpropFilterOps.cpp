// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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

REGISTER_OP("Open3DContinuousConvBackpropFilter")
        .Attr("TReal: {float, double}")
        .Attr("TIndex: {int32, int64}")
        .Attr("align_corners: bool = true")
        .Attr("coordinate_mapping: {'ball_to_cube_radial', "
              "'ball_to_cube_volume_preserving', 'identity'} = "
              "'ball_to_cube_radial'")
        .Attr("normalize: bool = false")
        .Attr("interpolation: {'linear', 'linear_border', 'nearest_neighbor'} "
              "= 'linear'")
        .Attr("max_temp_mem_MB: int = 64")
        .Attr("debug: bool = false")
        .Input("filters: TReal")        // [depth, height, width, in_ch, out_ch]
        .Input("out_positions: TReal")  // [num_points_out, 3]
        .Input("extents: TReal")        // [num_points_out, 3]
        .Input("offset: TReal")         // [3]
        .Input("inp_positions: TReal")  // [num_points_in, 3]
        .Input("inp_features: TReal")   // [num_points_in, in_ch]
        .Input("inp_importance: TReal")         // [num_points_in]
        .Input("neighbors_index: TIndex")       // [?]
        .Input("neighbors_importance: TReal")   // [?]
        .Input("neighbors_row_splits: int64")   // [num_points_out]
        .Input("out_features_gradient: TReal")  // [num_points_out, out_ch]
        .Output("filter_backprop : TReal")      // [depth, height, width, in_ch,
                                                // out_ch]
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            using namespace ::tensorflow::shape_inference;
            ShapeHandle filters_shape, out_positions_shape, extents_shape,
                    offset_shape, inp_positions_shape, inp_features_shape,
                    inp_importance_shape, neighbors_index_shape,
                    neighbors_importance_shape, neighbors_row_splits_shape,
                    out_features_gradient_shape;

            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 5, &filters_shape));
            TF_RETURN_IF_ERROR(
                    c->WithRank(c->input(1), 2, &out_positions_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &extents_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &offset_shape));
            TF_RETURN_IF_ERROR(
                    c->WithRank(c->input(4), 2, &inp_positions_shape));
            TF_RETURN_IF_ERROR(
                    c->WithRank(c->input(5), 2, &inp_features_shape));
            TF_RETURN_IF_ERROR(
                    c->WithRank(c->input(6), 1, &inp_importance_shape));
            TF_RETURN_IF_ERROR(
                    c->WithRank(c->input(7), 1, &neighbors_index_shape));
            TF_RETURN_IF_ERROR(
                    c->WithRank(c->input(8), 1, &neighbors_importance_shape));
            TF_RETURN_IF_ERROR(
                    c->WithRank(c->input(9), 1, &neighbors_row_splits_shape));
            TF_RETURN_IF_ERROR(
                    c->WithRank(c->input(10), 2, &out_features_gradient_shape));

            //
            // check if dimensions make sense between tensors
            //
            if (c->RankKnown(out_positions_shape) &&
                c->RankKnown(neighbors_row_splits_shape)) {
                DimensionHandle d;
                DimensionHandle num_out;
                TF_RETURN_IF_ERROR(c->Subtract(
                        c->Dim(neighbors_row_splits_shape, 0), 1, &num_out));
                TF_RETURN_IF_ERROR(
                        c->Merge(c->Dim(out_positions_shape, 0), num_out, &d));
            }

            if (c->RankKnown(inp_positions_shape) &&
                c->RankKnown(inp_features_shape)) {
                DimensionHandle d;
                TF_RETURN_IF_ERROR(c->Merge(c->Dim(inp_positions_shape, 0),
                                            c->Dim(inp_features_shape, 0), &d));
            }

            if (c->RankKnown(filters_shape) &&
                c->RankKnown(inp_features_shape)) {
                DimensionHandle d;
                TF_RETURN_IF_ERROR(c->Merge(c->Dim(filters_shape, 3),
                                            c->Dim(inp_features_shape, 1), &d));
            }

            if (c->RankKnown(extents_shape)) {
                DimensionHandle d;
                Status s1 = c->WithValue(c->Dim(extents_shape, 1), 1, &d);
                Status s2 = c->WithValue(c->Dim(extents_shape, 1), 3, &d);

                if (!s1.ok() && !s2.ok())
                    TF_RETURN_WITH_CONTEXT_IF_ERROR(
                            c->WithValue(c->Dim(extents_shape, 1), 3, &d),
                            "extents must have 3 components or 1 component");
            }

            if (c->RankKnown(offset_shape)) {
                DimensionHandle d;
                TF_RETURN_IF_ERROR(
                        c->WithValue(c->Dim(offset_shape, 0), 3, &d));
            }

            //
            // check if the filter shape is valid
            //
            for (int i = 0; i < 3; ++i) {
                if (c->ValueKnown(c->Dim(filters_shape, i))) {
                    int64_t n = c->Value(c->Dim(filters_shape, i));
                    if (n < 1)
                        return Status(error::INVALID_ARGUMENT,
                                      "Each filter dimension must be >= 1");
                }
            }

            // check shape of the incoming gradient 'out_features_gradient'
            if (c->RankKnown(out_features_gradient_shape)) {
                DimensionHandle first_dim = c->UnknownDim();
                if (c->RankKnown(out_positions_shape)) {
                    TF_RETURN_IF_ERROR(c->Merge(c->Dim(out_positions_shape, 0),
                                                first_dim, &first_dim));
                }

                DimensionHandle second_dim = c->UnknownDim();
                if (c->RankKnown(filters_shape)) {
                    TF_RETURN_IF_ERROR(c->Merge(c->Dim(filters_shape, 4),
                                                second_dim, &second_dim));
                }
                ShapeHandle features_grad_shape =
                        c->MakeShape({first_dim, second_dim});

                ShapeHandle s;
                TF_RETURN_IF_ERROR(c->Merge(features_grad_shape,
                                            out_features_gradient_shape, &s));
            }

            c->set_output(0, filters_shape);
            return Status::OK();
        })
        .Doc(R"doc(
Computes the backprop for the filter of the ContinuousConv

align_corners:
  If True the outer voxel centers of the filter grid are aligned with the boundady of the spatial shape.


coordinate_mapping:
  Defines how the relative positions of the neighbors are mapped before computing
  filter indices.
  For all mappings relative coordinates will be scaled with the inverse extent,
  i.e. the extent becomes a unit cube.
  After that one of the following mappings will be applied:
    'ball_to_cube_radial': maps a unit ball to a unit cube by radial stretching.
    'ball_to_cube_volume_preserving': maps a unit ball to a unit cube preserving the volume.
    'identity': the identity mapping.
  Use 'ball_to_cube_radial' for a spherical or ellipsoidal filter window
  and 'identiy' for a rectangular filter window.


normalize:
  If True the output feature values will be normalized by the number of neighbors.


interpolation:
  If interpolation is 'linear' then each filter value lookup is a trilinear interpolation.
  If interpolation is 'nearest_neighbor' only the spatially closest value is considered.
  This makes the filter and therefore the convolution discontinuous.


max_temp_mem_MB:
  Defines the maximum temporary memory in megabytes to be used for the GPU
  implementation. More memory means fewer kernel invocations. Note that the
  a minimum amount of temp memory will always be allocated even if this
  variable is set to 0.


filters:
  The filter parameters.
  The shape of the filter is [depth, height, width, in_ch, out_ch].
  The dimensions 'depth', 'height', 'width' define the spatial resolution of
  the filter. The spatial size of the filter is defined by the parameter
  'extents'.


out_positions:
  A 2D tensor with the 3D point positions of each output point.
  The coordinates for each point is a vector with format [x,y,z].


extents:
  The extent defines the spatial size of the filter for each output point.
  It is a 2D vector of the form [[x_size, y_size, z_size], ..].
  For 'ball to cube' coordinate mappings the extent defines the bounding box
  of the ball.
  Broadcasting is supported for all axes. E.g. providing only the extent for a
  single point as well as only providing 'x_size' is valid.


offset:
  A 1D tensor which defines the offset in voxel units to shift the input points.
  Offsets will be ignored if align_corners is True.


inp_positions:
  A 2D tensor with the 3D point positions of each input point.
  The coordinates for each point is a vector with format [x,y,z].


inp_features:
  A 2D tensor which stores a feature vector for each input point.


neighbors_index:
  The neighbors_index stores a list of indices of neighbors for each output point as nested lists.
  The start and end of each list can be computed using 'neighbors_row_splits'.


neighbors_importance:
  Tensor of the same shape as 'neighbors_index' with a scalar value that is used to scale
  the features of each neighbor.


neighbors_row_splits:
  The exclusive prefix sum of the neighbor count for the output points including
  the total neighbor count as the last element. The size of this array is the
  number of output points + 1.


out_features_gradient:
  A Tensor with the gradient for the outputs of the DCConv in the forward pass.


filter_backprop:
  The gradients for the filter

)doc");
