// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/ml/tensorflow/TensorFlowHelper.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"

using namespace tensorflow;

REGISTER_OP("Open3DKnnSearch")
        .Attr("T: {float, double}")
        .Attr("index_dtype: {int32, int64} = DT_INT32")
        .Attr("metric: {'L1', 'L2'} = 'L2'")
        .Attr("ignore_query_point: bool = false")
        .Attr("return_distances: bool = false")
        .Input("points: T")
        .Input("queries: T")
        .Input("k: int32")
        .Input("points_row_splits: int64")
        .Input("queries_row_splits: int64")
        .Output("neighbors_index: index_dtype")
        .Output("neighbors_row_splits: int64")
        .Output("neighbors_distance: T")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            using namespace ::tensorflow::shape_inference;
            using namespace open3d::ml::op_util;
            ShapeHandle points_shape, queries_shape, k_shape,
                    points_row_splits_shape, queries_row_splits_shape,
                    neighbors_index_shape, neighbors_row_splits_shape,
                    neighbors_distance_shape;

            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &points_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &queries_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &k_shape));
            TF_RETURN_IF_ERROR(
                    c->WithRank(c->input(3), 1, &points_row_splits_shape));
            TF_RETURN_IF_ERROR(
                    c->WithRank(c->input(4), 1, &queries_row_splits_shape));

            Dim num_points("num_points");
            Dim num_queries("num_queries");
            Dim batch_size("batch_size");
            CHECK_SHAPE_HANDLE(c, points_shape, num_points, 3);
            CHECK_SHAPE_HANDLE(c, queries_shape, num_queries, 3);
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

            return Status();
        })
        .Doc(R"doc(
Computes the indices of k nearest neighbors.

This op computes the neighborhood for each query point and returns the indices
of the neighbors. The output format is compatible with the radius_search and
fixed_radius_search ops and supports returning less than k neighbors if there
are less than k points or ignore_query_point is enabled and the **queries** and
**points** arrays are the same point cloud. The following example shows the usual
case where the outputs can be reshaped to a [num_queries, k] tensor::

  import tensorflow as tf
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

  ans = ml3d.ops.knn_search(points, queries, k=2,
                      points_row_splits=[0,5],
                      queries_row_splits=[0,3],
                      return_distances=True)
  # returns ans.neighbors_index      = [1, 2, 4, 2, 4, 2]
  #         ans.neighbors_row_splits = [0, 2, 4, 6]
  #         ans.neighbors_distance   = [0.75 , 1.47, 0.56, 1.62, 0.77, 1.85]
  # Since there are more than k points and we do not ignore any points we can
  # reshape the output to [num_queries, k] with
  neighbors_index = tf.reshape(ans.neighbors_index, [3,2])
  neighbors_distance = tf.reshape(ans.neighbors_distance, [3,2])


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
      [0.5,2.1,2.2],
  ])

  radii = torch.Tensor([1.0,1.0,1.0])

  ans = ml3d.ops.knn_search(points, queries, k=2,
                            points_row_splits=torch.LongTensor([0,5]),
                            queries_row_splits=torch.LongTensor([0,3]),
                            return_distances=True)
  # returns ans.neighbors_index      = [1, 2, 4, 2, 4, 2]
  #         ans.neighbors_row_splits = [0, 2, 4, 6]
  #         ans.neighbors_distance   = [0.75 , 1.47, 0.56, 1.62, 0.77, 1.85]
  # Since there are more than k points and we do not ignore any points we can
  # reshape the output to [num_queries, k] with
  neighbors_index = ans.neighbors_index.reshape(3,2)
  neighbors_distance = ans.neighbors_distance.reshape(3,2)

metric: Either L1 or L2. Default is L2

ignore_query_point: If true the points that coincide with the center of the
  search window will be ignored. This excludes the query point if **queries** and
 **points** are the same point cloud.

return_distances: If True the distances for each neighbor will be returned in
  the output tensor **neighbors_distances**. If False a zero length Tensor will
  be returned for **neighbors_distances**.

points: The 3D positions of the input points.

queries: The 3D positions of the query points.

k: The number of nearest neighbors to search.

points_row_splits: 1D vector with the row splits information if points is
  batched. This vector is [0, num_points] if there is only 1 batch item.

queries_row_splits: 1D vector with the row splits information if queries is
  batched. This vector is [0, num_queries] if there is only 1 batch item.

neighbors_index: The compact list of indices of the neighbors. The
  corresponding query point can be inferred from the
  **neighbor_count_prefix_sum** vector. Neighbors for the same point are sorted
  with respect to the distance.

  Note that there is no guarantee that there will be exactly k neighbors in some cases.
  These cases are:
    * There are less than k points.
    * **ignore_query_point** is True and there are multiple points with the same position.

neighbors_row_splits: The exclusive prefix sum of the neighbor count for the
  query points including the total neighbor count as the last element. The
  size of this array is the number of queries + 1.

neighbors_distance: Stores the distance to each neighbor if **return_distances**
  is True. The distances are squared only if metric is L2. This is a zero length
  Tensor if **return_distances** is False.

)doc");
