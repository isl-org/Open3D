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

REGISTER_OP("Open3DFixedRadiusSearch")
        .Attr("T: {float, double}")
        .Attr("metric: {'L1', 'L2', 'Linf'} = 'L2'")
        .Attr("ignore_query_point: bool = false")
        .Attr("return_distances: bool = false")
        .Attr("max_hash_table_size: int = 33554432")
        .Input("points: T")
        .Input("queries: T")
        .Input("radius: T")
        .Input("hash_table_size_factor: double")
        .Output("neighbors_index: int32")
        .Output("neighbors_row_splits: int64")
        .Output("neighbors_distance: T")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            using namespace ::tensorflow::shape_inference;
            ShapeHandle points_shape, queries_shape, radius_shape,
                    hash_table_size_factor_shape, indices_shape,
                    neighbors_row_splits_shape, neighbors_distance_shape;

            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &points_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &queries_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &radius_shape));
            TF_RETURN_IF_ERROR(
                    c->WithRank(c->input(3), 0, &hash_table_size_factor_shape));

            // check if we have [N,3] tensors for the positions
            if (c->RankKnown(points_shape)) {
                DimensionHandle d;
                TF_RETURN_IF_ERROR(
                        c->WithValue(c->Dim(points_shape, -1), 3, &d));
            }

            DimensionHandle num_query_points = c->UnknownDim();
            if (c->RankKnown(queries_shape)) {
                num_query_points = c->Dim(queries_shape, 0);

                DimensionHandle d;
                TF_RETURN_IF_ERROR(
                        c->WithValue(c->Dim(queries_shape, -1), 3, &d));
            }

            // we cannot infer the number of neighbors
            indices_shape = c->MakeShape({c->UnknownDim()});
            c->set_output(0, indices_shape);

            DimensionHandle neighbors_row_splits_size;
            TF_RETURN_IF_ERROR(
                    c->Add(num_query_points, 1, &neighbors_row_splits_size));
            neighbors_row_splits_shape =
                    c->MakeShape({neighbors_row_splits_size});
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
of the neighbors.

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

max_hash_table_size:
  The maximum hash table size.

points: 
  The 3D positions of the input points.

queries: 
  The 3D positions of the query points.

radius:
  A scalar with the neighborhood radius

hash_table_size_factor:
  The size of the hash table as a factor of the number of input points.

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
