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

REGISTER_OP("Open3DFixedRadiusSearch")
        .Attr("T: {float, double}")
        .Attr("metric: {'L1', 'L2', 'Linf'} = 'L2'")
        .Attr("ignore_query_point: bool = false")
        .Attr("return_distances: bool = false")
        .Input("points: T")
        .Input("queries: T")
        .Input("radius: T")
        .Input("points_row_splits: int64")
        .Input("queries_row_splits: int64")
        .Input("hash_table_splits: uint32")
        .Input("hash_table_index: uint32")
        .Input("hash_table_cell_splits: uint32")
        .Output("neighbors_index: int32")
        .Output("neighbors_row_splits: int64")
        .Output("neighbors_distance: T")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            using namespace ::tensorflow::shape_inference;
            using namespace open3d::ml::op_util;
            ShapeHandle points_shape, queries_shape, radius_shape,
                    points_row_splits_shape, queries_row_splits_shape,
                    hash_table_splits_shape, hash_table_index_shape,
                    hash_table_row_splits_shape, neighbors_index_shape,
                    neighbors_row_splits_shape, neighbors_distance_shape;

            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &points_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &queries_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &radius_shape));
            TF_RETURN_IF_ERROR(
                    c->WithRank(c->input(3), 1, &points_row_splits_shape));
            TF_RETURN_IF_ERROR(
                    c->WithRank(c->input(4), 1, &queries_row_splits_shape));
            TF_RETURN_IF_ERROR(
                    c->WithRank(c->input(5), 1, &hash_table_splits_shape));
            TF_RETURN_IF_ERROR(
                    c->WithRank(c->input(6), 1, &hash_table_index_shape));
            TF_RETURN_IF_ERROR(
                    c->WithRank(c->input(7), 1, &hash_table_row_splits_shape));

            Dim num_points("num_points");
            Dim num_queries("num_queries");
            Dim batch_size("batch_size");
            CHECK_SHAPE_HANDLE(c, points_shape, num_points, 3);
            CHECK_SHAPE_HANDLE(c, hash_table_index_shape, num_points);
            CHECK_SHAPE_HANDLE(c, queries_shape, num_queries, 3);
            CHECK_SHAPE_HANDLE(c, points_row_splits_shape, batch_size + 1);
            CHECK_SHAPE_HANDLE(c, queries_row_splits_shape, batch_size + 1);
            CHECK_SHAPE_HANDLE(c, hash_table_splits_shape, batch_size + 1);

            // we cannot infer the number of neighbors
            neighbors_index_shape = c->MakeShape({c->UnknownDim()});
            c->set_output(0, neighbors_index_shape);

            neighbors_row_splits_shape = MakeShapeHandle(c, num_queries + 1);
            c->set_output(1, neighbors_row_splits_shape);

            bool return_distances;
            TF_RETURN_IF_ERROR(
                    c->GetAttr("return_distances", &return_distances));
            if (return_distances)
                neighbors_distance_shape = c->MakeShape({c->UnknownDim()});
            else
                neighbors_distance_shape = c->MakeShape({0});
            c->set_output(2, neighbors_distance_shape);

            return Status::OK();
        })
        .Doc(R"doc(
Computes the indices of all neighbors within a radius.

This op computes the neighborhood for each query point and returns the indices
of the neighbors and optionally also the distances. The same fixed radius is
used for each query point. Points and queries can be batched with each batch 
item having an individual number of points and queries. The following example
shows a simple search with just a single batch item::


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


metric:
  Either L1, L2 or Linf. Default is L2

ignore_query_point:
  If true the points that coincide with the center of the search window will be
  ignored. This excludes the query point if 'queries' and 'points' are the same
  point cloud.

return_distances:
  If True the distances for each neighbor will be returned in the tensor
  'neighbors_distance'.
  If False a zero length Tensor will be returned for 'neighbors_distance'.

points:
  The 3D positions of the input points.

queries:
  The 3D positions of the query points.

radius:
  A scalar with the neighborhood radius

points_row_splits:
  1D vector with the row splits information if points is batched.
  This vector is [0, num_points] if there is only 1 batch item.

queries_row_splits:
  1D vector with the row splits information if queries is batched.
  This vector is [0, num_queries] if there is only 1 batch item.

hash_table_splits: Array defining the start and end the hash table
  for each batch item. This is [0, number of cells] if there is only
  1 batch item or [0, hash_table_cell_splits_size-1] which is the same.

hash_table_index: Stores the values of the hash table, which are the indices of
  the points. The start and end of each cell is defined by hash_table_cell_splits.

hash_table_cell_splits: Defines the start and end of each hash table cell.

neighbors_index:
  The compact list of indices of the neighbors. The corresponding query point
  can be inferred from the 'neighbor_count_row_splits' vector.

neighbors_row_splits:
  The exclusive prefix sum of the neighbor count for the query points including
  the total neighbor count as the last element. The size of this array is the
  number of queries + 1.

neighbors_distance:
  Stores the distance to each neighbor if 'return_distances' is True.
  Note that the distances are squared if metric is L2.
  This is a zero length Tensor if 'return_distances' is False.

)doc");
