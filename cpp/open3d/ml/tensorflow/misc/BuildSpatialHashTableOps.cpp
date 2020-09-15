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

#include "open3d/ml/tensorflow/TensorFlowHelper.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"

using namespace tensorflow;

REGISTER_OP("Open3DBuildSpatialHashTable")
        .Attr("T: {float, double}")
        .Attr("max_hash_table_size: int = 33554432")
        .Input("points: T")
        .Input("radius: T")
        .Input("points_row_splits: int64")
        .Input("hash_table_size_factor: double")
        .Output("hash_table_index: uint32")
        .Output("hash_table_cell_splits: uint32")
        .Output("hash_table_splits: uint32")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            using namespace ::tensorflow::shape_inference;
            using namespace open3d::ml::op_util;
            ShapeHandle points_shape, radius_shape, points_row_splits_shape,
                    hash_table_size_factor_shape, hash_table_index_shape,
                    hash_table_cell_splits_shape, hash_table_splits_shape;

            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &points_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &radius_shape));
            TF_RETURN_IF_ERROR(
                    c->WithRank(c->input(2), 1, &points_row_splits_shape));
            TF_RETURN_IF_ERROR(
                    c->WithRank(c->input(3), 0, &hash_table_size_factor_shape));

            // check if we have [N,3] tensors for the positions
            Dim num_points("num_points");
            Dim batch_size("batch_size");
            CHECK_SHAPE_HANDLE(c, points_shape, num_points, 3);
            CHECK_SHAPE_HANDLE(c, points_row_splits_shape, batch_size + 1);

            hash_table_index_shape = MakeShapeHandle(c, num_points);
            c->set_output(0, hash_table_index_shape);

            hash_table_cell_splits_shape = c->MakeShape({c->UnknownDim()});
            c->set_output(1, hash_table_cell_splits_shape);

            hash_table_splits_shape = MakeShapeHandle(c, batch_size + 1);
            c->set_output(2, hash_table_splits_shape);

            return Status::OK();
        })
        .Doc(R"doc(
Creates a spatial hash table meant as input for fixed_radius_search


The following example shows how **build_spatial_hash_table** and 
**fixed_radius_search** are used together::

  import open3d.ml.tf as ml3d

  points = [
    [0.1,0.1,0.1], 
    [0.5,0.5,0.5], 
    [1.7,1.7,1.7],
    [1.8,1.8,1.8],
    [0.3,2.4,1.4]]

  queries = [
      [1.0,1.0,1.0],
      [0.5,2.0,2.0],
      [0.5,2.1,2.1],
  ]

  radius = 1.0

  # build the spatial hash table for fixex_radius_search
  table = ml3d.ops.build_spatial_hash_table(points, 
                                            radius, 
                                            points_row_splits=torch.LongTensor([0,5]), 
                                            hash_table_size_factor=1/32)

  # now run the fixed radius search
  ml3d.ops.fixed_radius_search(points, 
                               queries, 
                               radius, 
                               points_row_splits=torch.LongTensor([0,5]), 
                               queries_row_splits=torch.LongTensor([0,3]), 
                               **table._asdict())
  # returns neighbors_index      = [1, 4, 4]
  #         neighbors_row_splits = [0, 1, 2, 3]
  #         neighbors_distance   = []

  # or with pytorch
  import torch
  import open3d.ml.torch as ml3d

  points = torch.Tensor([
    [0.1,0.1,0.1], 
    [0.5,0.5,0.5], 
    [1.7,1.7,1.7],
    [1.8,1.8,1.8],
    [0.3,2.4,1.4]])

  queries = torch.Tensor([
      [1.0,1.0,1.0],
      [0.5,2.0,2.0],
      [0.5,2.1,2.1],
  ])

  radius = 1.0

  # build the spatial hash table for fixex_radius_search
  table = ml3d.ops.build_spatial_hash_table(points, 
                                            radius, 
                                            points_row_splits=torch.LongTensor([0,5]), 
                                            hash_table_size_factor=1/32)

  # now run the fixed radius search
  ml3d.ops.fixed_radius_search(points, 
                               queries, 
                               radius, 
                               points_row_splits=torch.LongTensor([0,5]), 
                               queries_row_splits=torch.LongTensor([0,3]), 
                               **table._asdict())
  # returns neighbors_index      = [1, 4, 4]
  #         neighbors_row_splits = [0, 1, 2, 3]
  #         neighbors_distance   = []



max_hash_table_size: The maximum hash table size.

points: The 3D positions of the input points.

radius: A scalar which defines the spatial cell size of the hash table.

points_row_splits: 1D vector with the row splits information if points is 
  batched. This vector is [0, num_points] if there is only 1 batch item.

hash_table_size_factor:
  The size of the hash table as a factor of the number of input points.

hash_table_index: Stores the values of the hash table, which are the indices of
  the points. The start and end of each cell is defined by 
  **hash_table_cell_splits**.

hash_table_cell_splits: Defines the start and end of each hash table cell within
  a hash table.

hash_table_splits: Defines the start and end of each hash table in the
  hash_table_cell_splits array. If the batch size is 1 then there is only one
  hash table and this vector is [0, number of cells].

)doc");
