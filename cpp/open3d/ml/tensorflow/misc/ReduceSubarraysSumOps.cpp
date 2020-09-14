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

REGISTER_OP("Open3DReduceSubarraysSum")
        .Attr("T: {int32, int64, float, double}")
        .Input("values: T")
        .Input("row_splits: int64")
        .Output("sums: T")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            using namespace ::tensorflow::shape_inference;
            ShapeHandle values_shape, row_splits_shape, sums_shape;

            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &values_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &row_splits_shape));

            // output will have one element less than row_splits
            DimensionHandle sums_size = c->UnknownDim();
            if (c->RankKnown(row_splits_shape)) {
                TF_RETURN_IF_ERROR(c->Subtract(c->Dim(row_splits_shape, 0), 1,
                                               &sums_size));
            }
            sums_shape = c->MakeShape({sums_size});
            c->set_output(0, sums_shape);

            return Status::OK();
        })
        .Doc(R"doc(
Computes the sum for each subarray in a flat vector of arrays.

The start and end of the subarrays are defined by an exclusive prefix sum.
Zero length subarrays are allowed as shown in the following example::

  import open3d.ml.tf as ml3d

  ml3d.ops.reduce_subarrays_sum(
      values = [1,2,3,4],
      row_splits=[0,2,2,4] # defines 3 subarrays with starts and ends 0-2,2-2,2-4
      )
  # returns [3,0,7]


  # or with pytorch
  import torch
  import open3d.ml.torch as ml3d
  
  ml3d.ops.reduce_subarrays_sum(
    values = torch.Tensor([1,2,3,4]),
    row_splits=torch.LongTensor([0,2,2,4]) # defines 3 subarrays with starts and ends 0-2,2-2,2-4
    )
  # returns [3,0,7]


values: Linear memory which stores the values for all arrays.

row_splits: Defines the start and end of each subarray. This is an exclusive
  prefix sum with 0 as the first element and the length of values as
  additional last element. If there are N subarrays the length of this vector
  is N+1.

sums: The sum of each subarray. The sum of an empty subarray is 0.
  sums is a zero length vector if values is a zero length vector.

)doc");
