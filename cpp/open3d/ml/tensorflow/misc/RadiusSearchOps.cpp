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

#include "open3d/ml/tensorflow/TensorFlowHelper.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"

using namespace tensorflow;

REGISTER_OP("Open3DRadiusSearch")
        .Attr("T: {float, double}")
        .Attr("metric: {'L1', 'L2'} = 'L2'")
        .Attr("ignore_query_point: bool = false")
        .Attr("return_distances: bool = false")
        .Attr("normalize_distances: bool = false")
        .Input("points: T")
        .Input("queries: T")
        .Input("radii: T")
        .Input("points_row_splits: int64")
        .Input("queries_row_splits: int64")
        .Output("neighbors_index: int32")
        .Output("neighbors_row_splits: int64")
        .Output("neighbors_distance: T")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            using namespace ::tensorflow::shape_inference;
            using namespace open3d::ml::op_util;
            ShapeHandle points_shape, queries_shape, radii_shape,
                    points_row_splits_shape, queries_row_splits_shape,
                    neighbors_index_shape, neighbors_row_splits_shape,
                    neighbors_distance_shape;

            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &points_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &queries_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &radii_shape));
            TF_RETURN_IF_ERROR(
                    c->WithRank(c->input(3), 1, &points_row_splits_shape));
            TF_RETURN_IF_ERROR(
                    c->WithRank(c->input(4), 1, &queries_row_splits_shape));

            Dim num_points("num_points");
            Dim num_queries("num_queries");
            Dim batch_size("batch_size");
            CHECK_SHAPE_HANDLE(c, points_shape, num_points, 3);
            CHECK_SHAPE_HANDLE(c, queries_shape, num_queries, 3);
            CHECK_SHAPE_HANDLE(c, radii_shape, num_queries);
            CHECK_SHAPE_HANDLE(c, points_row_splits_shape, batch_size + 1);
            CHECK_SHAPE_HANDLE(c, queries_row_splits_shape, batch_size + 1);

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
Computes the indices and distances of all neigbours within a radius.

This op computes the neighborhood for each query point and returns the indices
of the neighbors and optionally also the distances. Each query point has an 
individual search radius. Points and queries can be batched with each batch 
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
      [0.5,2.1,2.2],
  ]

  radii = [1.0,1.0,1.0]

  ml3d.ops.radius_search(points, queries, radii, 
                         points_row_splits=[0,5], 
                         queries_row_splits=[0,3]) 
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

  radii = torch.Tensor([1.0,1.0,1.0])

  ml3d.ops.radius_search(points, queries, radii, 
                         points_row_splits=torch.LongTensor([0,5]), 
                         queries_row_splits=torch.LongTensor([0,3]))
  # returns neighbors_index      = [1, 4, 4]
  #         neighbors_row_splits = [0, 1, 2, 3]
  #         neighbors_distance   = []


metric: Either L1 or L2. Default is L2

ignore_query_point: If true the points that coincide with the center of the 
  search window will be ignored. This excludes the query point if **queries** and 
  **points** are the same point cloud.

return_distances: If True the distances for each neighbor will be returned in 
  the output tensor **neighbors_distance**.  If False a zero length Tensor will 
  be returned for **neighbors_distances**.

normalize_distances: If True the returned distances will be normalized with the
  radii.

points: The 3D positions of the input points.

queries: The 3D positions of the query points.

radii: A vector with the individual radii for each query point.

points_row_splits: 1D vector with the row splits information if points is 
  batched. This vector is [0, num_points] if there is only 1 batch item.

queries_row_splits: 1D vector with the row splits information if queries is 
  batched. This vector is [0, num_queries] if there is only 1 batch item.

neighbors_index: The compact list of indices of the neighbors. The 
  corresponding query point can be inferred from the 
  **neighbor_count_row_splits** vector.

neighbors_row_splits: The exclusive prefix sum of the neighbor count for the 
  query points including the total neighbor count as the last element. The 
  size of this array is the number of queries + 1.

neighbors_distance: Stores the distance to each neighbor if **return_distances** 
  is True. The distances are squared only if metric is L2.
  This is a zero length Tensor if **return_distances** is False.

)doc");
