// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/lib/core/errors.h>

#include "open3d/ml/tensorflow/TensorFlowHelper.h"

using namespace tensorflow;

REGISTER_OP("Open3DSparseConvTransposeBackpropFilter")
        .Attr("TFeat: {float, double, bfloat16}")  // Type for features and
                                                   // weights
        .Attr("output_type: {float, double, bfloat16} = DT_FLOAT")  // Type for
                                                                    // the
                                                                    // output
                                                                    // features
        .Attr("TIndex: {int32, int64}")
        .Attr("TKernelIndex: {uint8, int16}")
        .Attr("normalize: bool = false")
        .Attr("max_temp_mem_MB: int = 64")
        .Input("filters: TFeat")  // [depth, height, width, in_ch, out_ch]
        .Input("out_importance: TFeat")                // [num_points_out]
        .Input("inp_features: TFeat")                  // [num_points_in, in_ch]
        .Input("inp_neighbors_importance_sum: TFeat")  // [num_points_in]
        .Input("inp_neighbors_row_splits: int64")      // [num_points_in]
        .Input("neighbors_index: TIndex")              // [?]
        .Input("neighbors_kernel_index: TKernelIndex")  // [?]
        .Input("neighbors_importance: TFeat")           // [?]
        .Input("neighbors_row_splits: int64")           // [num_points_out]
        .Input("out_features_gradient: TFeat")    // [num_points_out, out_ch]
        .Output("filter_backprop : output_type")  // [depth, height, width,
                                                  // in_ch, out_ch]
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            using namespace ::tensorflow::shape_inference;
            using namespace open3d::ml::op_util;
            ShapeHandle filters = c->input(0);
            ShapeHandle out_importance = c->input(1);
            ShapeHandle inp_features = c->input(2);
            ShapeHandle inp_neighbors_importance_sum = c->input(3);
            ShapeHandle inp_neighbors_row_splits = c->input(4);
            ShapeHandle neighbors_index = c->input(5);
            ShapeHandle neighbors_kernel_index = c->input(6);
            ShapeHandle neighbors_importance = c->input(7);
            ShapeHandle neighbors_row_splits = c->input(8);
            ShapeHandle out_features_gradient = c->input(9);

            Dim num_out("num_out");
            Dim num_inp("num_inp");
            Dim num_kernel_elements("num_kernel_elements");
            Dim in_channels("in_channels");
            Dim out_channels("out_channels");
            Dim num_neighbors("num_neighbors");

            CHECK_SHAPE_HANDLE_COMBINE_FIRST_DIMS(
                    c, filters, num_kernel_elements, in_channels, out_channels);
            CHECK_SHAPE_HANDLE(c, neighbors_row_splits, num_out + 1);
            CHECK_SHAPE_HANDLE(c, out_importance, 0 || num_out);
            CHECK_SHAPE_HANDLE(c, inp_features, num_inp, in_channels);
            CHECK_SHAPE_HANDLE(c, inp_neighbors_importance_sum, 0 || num_inp);
            CHECK_SHAPE_HANDLE(c, inp_neighbors_row_splits, num_inp + 1);
            CHECK_SHAPE_HANDLE(c, neighbors_index, num_neighbors);
            CHECK_SHAPE_HANDLE(c, neighbors_kernel_index, num_neighbors);
            CHECK_SHAPE_HANDLE(c, neighbors_importance, 0 || num_neighbors);
            CHECK_SHAPE_HANDLE(c, out_features_gradient, num_out, out_channels);

            c->set_output(0, filters);
            return Status();
        })
        .Doc(R"doc(
Computes the backrop for the filter of the SparseConvTranspose

normalize:
  If True the input feature values will be normalized by the number of neighbors.


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


inp_features:
  A 2D tensor which stores a feature vector for each input point.


inp_neighbors_row_splits:
  The number of neighbors for each input point as exclusive prefix sum.


neighbors_index:
  The neighbors_index stores a list of indices of neighbors for each output point as nested lists.
  The start and end of each list can be computed using 'neighbors_row_splits'.


neighbors_kernel_index:
  Defines which kernel element to use for each neighbor. This array has the same length as neighbors_index.


neighbors_row_splits:
  The number of neighbors for each output point as exclusive prefix sum.


out_features_gradient:
  A Tensor with the gradient for the outputs of the SparseConvTranspose in the forward pass.

output_type: The type for the output.

filter_backprop:
  The gradients for the filter

)doc");
