// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 www.open3d.org
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

REGISTER_OP("Open3DInvertNeighborsList")
        .Attr("TIndex: {int32}")
        .Attr("TAttr: {int32, int64, float, double}")
        .Input("num_points: int64")
        .Input("inp_neighbors_index: TIndex")
        .Input("inp_neighbors_prefix_sum: int64")
        .Input("inp_neighbors_attributes: TAttr")
        .Output("neighbors_index: TIndex")
        .Output("neighbors_prefix_sum: int64")
        .Output("neighbors_attributes: TAttr")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            using namespace ::tensorflow::shape_inference;
            ShapeHandle num_points_shape, inp_neighbors_index_shape,
                    inp_neighbors_prefix_sum_shape,
                    inp_neighbors_attributes_shape, neighbors_index_shape,
                    neighbors_count_prefix_sum_shape;

            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &num_points_shape));
            TF_RETURN_IF_ERROR(
                    c->WithRank(c->input(1), 1, &inp_neighbors_index_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1,
                                           &inp_neighbors_prefix_sum_shape));
            TF_RETURN_IF_ERROR(c->WithRankAtLeast(
                    c->input(3), 1, &inp_neighbors_attributes_shape));

            // output will have the same shape
            c->set_output(0, inp_neighbors_index_shape);

            // The length of the prefix sum vector will change to num_points
            // which we do not know here
            neighbors_count_prefix_sum_shape = c->MakeShape({c->UnknownDim()});
            c->set_output(1, neighbors_count_prefix_sum_shape);

            // the attributes will have the same shape
            c->set_output(2, inp_neighbors_attributes_shape);
            return Status::OK();
        })
        .Doc(R"doc(
Inverts a neighbors list stored as neighbor indices and prefix sum and returns 
the inverted list in the same format.


num_points:
  Scalar integer with the number of points that have been tested in a neighbor 
  search. This is the number of the points in the second point cloud (not the
  query point cloud) in a neighbor search.
  This will be the size of the output prefix_sum.

inp_neighbors_index:
  The input neighbor indices stored linearly.

inp_neighbors_prefix_sum:
  The number of neighbors for the input queries as exclusive prefix sum.

inp_neighbors_attributes:
  Array that stores an attribute for each neighbor. 
  The size of the first dim must match the first dim of inp_neighbors_index.
  To ignore attributes pass a 1D Tensor with zero size.

neighbors_index: 
  The output neighbor indices stored linearly.

neighbors_prefix_sum:
  Stores the number of neighbors for the new queries (previously the input 
  points) as exclusive prefix sum.

neighbors_attributes:
  Array that stores an attribute for each neighbor. 
  If the inp_neighbors_attributes Tensor is a zero length vector then the output
  will be a zero length vector as well. 

)doc");
