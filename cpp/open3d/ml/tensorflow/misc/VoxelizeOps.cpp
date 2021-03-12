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

REGISTER_OP("Open3DVoxelize")
        .Attr("T: {float, double}")  // type for the point positions
        .Attr("max_points_per_voxel: int = 9223372036854775807")
        .Attr("max_voxels: int = 9223372036854775807")
        .Input("points: T")
        .Input("voxel_size: T")
        .Input("points_range_min: T")
        .Input("points_range_max: T")
        .Output("voxel_coords: int32")
        .Output("voxel_point_indices: int64")
        .Output("voxel_point_row_splits: int64")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            using namespace ::tensorflow::shape_inference;
            using namespace open3d::ml::op_util;
            ShapeHandle points, voxel_size, points_range_min, points_range_max,
                    max_points_per_voxel, max_voxels, voxel_coords,
                    voxel_point_indices, voxel_point_row_splits;

            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &points));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &voxel_size));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &points_range_min));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &points_range_max));

            Dim num_points("num_points");
            Dim ndim("ndim");
            CHECK_SHAPE_HANDLE(c, points, num_points, ndim);
            CHECK_SHAPE_HANDLE(c, voxel_size, ndim);
            CHECK_SHAPE_HANDLE(c, points_range_min, ndim);
            CHECK_SHAPE_HANDLE(c, points_range_max, ndim);

            voxel_coords =
                    c->MakeShape({c->UnknownDim(), c->MakeDim(ndim.value())});
            c->set_output(0, voxel_coords);

            voxel_point_indices = c->MakeShape({c->UnknownDim()});
            c->set_output(1, voxel_point_indices);
            voxel_point_row_splits = c->MakeShape({c->UnknownDim()});
            c->set_output(2, voxel_point_row_splits);

            return Status::OK();
        })
        .Doc(R"doc(
Voxelization for point clouds.

The function returns the integer coordinates of the voxels that contain
points and a compact list of the indices that associate the voxels to the
points.

Minimal example::

  import open3d.ml.tf as ml3d

  points = [
      [0.1,0.1,0.1], 
      [0.5,0.5,0.5], 
      [1.7,1.7,1.7],
      [1.8,1.8,1.8],
      [9.3,9.4,9.4]]

  ml3d.ops.voxelize(points, 
                    voxel_size=[1.0,1.0,1.0], 
                    points_range_min=[0,0,0], 
                    points_range_max=[2,2,2])

  # returns the voxel coordinates  [[0, 0, 0],
  #                                 [1, 1, 1]]
  #
  #         the point indices      [0, 1, 2, 3]
  #
  #         and the point row splits [0, 2, 4] 

  # or with pytorch
  import torch
  import open3d.ml.torch as ml3d

  points = torch.Tensor([
      [0.1,0.1,0.1], 
      [0.5,0.5,0.5], 
      [1.7,1.7,1.7],
      [1.8,1.8,1.8],
      [9.3,9.4,9.4]])

  ml3d.ops.voxelize(points, 
                    voxel_size=torch.Tensor([1.0,1.0,1.0]), 
                    points_range_min=torch.Tensor([0,0,0]), 
                    points_range_max=torch.Tensor([2,2,2]))

  # returns the voxel coordinates  [[0, 0, 0],
  #                                 [1, 1, 1]]
  #
  #         the point indices      [0, 1, 2, 3]
  #
  #         and the point row splits [0, 2, 4] 

points: The point positions with shape [N,D] with N as the number of points and
  D as the number of dimensions, which must be 0 < D < 9.

voxel_size: The voxel size with shape [D].

points_range_min: The minimum range for valid points to be voxelized. This 
  vector has shape [D] and is used as the origin for computing the voxel_indices.

points_range_min: The maximum range for valid points to be voxelized. This 
  vector has shape [D].

max_points_per_voxel: The maximum number of points to consider for a voxel.

max_voxels: The maximum number of voxels to generate.

voxel_coords: The integer voxel coordinates.The shape of this tensor is [M, D]
  with M as the number of voxels and D as the number of dimensions.

voxel_point_indices: A flat list of all the points that have been voxelized.
  The start and end of each voxel is defined in voxel_point_row_splits.

voxel_point_row_splits: This is an exclusive prefix sum that includes the total
  number of points in the last element. This can be used to find the start and
  end of the point indices for each voxel. The shape of this tensor is [M+1].

)doc");
