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
        .Input("prefix_sum: int64")
        .Output("sums: T")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            using namespace ::tensorflow::shape_inference;
            ShapeHandle values_shape, prefix_sum_shape, sums_shape;

            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &values_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &prefix_sum_shape));

            // output will have the same shape as the prefix_sum
            c->set_output(0, prefix_sum_shape);

            return Status::OK();
        })
        .Doc(R"doc(
Computes the sum for each subarray. The start and end of the subarrays are defined by a prefix sum.


values:
  Linear memory which stores the values for all arrays.

prefix_sum:
  The prefix sum defines the start of each subarray.
  The end is defined by the next entry in the prefix_sum or the end of the 
  values array.

sums:
  The sum of each subarray. The sum of an empty subarray is 0.
  sums is a zero length vector if values is a zero length vector.

)doc");
