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

REGISTER_OP("Open3DNms")
        .Attr("T: {float}")  // type for boxes and scores
        .Attr("nms_overlap_thresh: float")
        .Input("boxes: T")
        .Input("scores: T")
        .Output("keep_indices: int64")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            using namespace ::tensorflow::shape_inference;
            using namespace open3d::ml::op_util;
            ShapeHandle boxes, scores, keep_indices;

            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &boxes));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &scores));

            Dim num_points("num_points");
            Dim five(5, "five");
            CHECK_SHAPE_HANDLE(c, boxes, num_points, five);
            CHECK_SHAPE_HANDLE(c, scores, num_points);

            keep_indices = c->MakeShape({c->UnknownDim()});
            c->set_output(0, keep_indices);
            return Status::OK();
        })
        .Doc(R"doc(
Performs non-maximum suppression of bounding boxes and returns the selected box
indices.

  # TensorFlow example.
  import open3d.ml.tf as ml3d
  import numpy as np

  boxes = np.array([[15.0811, -7.9803, 15.6721, -6.8714, 0.5152],
                    [15.1166, -7.9261, 15.7060, -6.8137, 0.6501],
                    [15.1304, -7.8129, 15.7069, -6.8903, 0.7296],
                    [15.2050, -7.8447, 15.8311, -6.7437, 1.0506],
                    [15.1343, -7.8136, 15.7121, -6.8479, 1.0352],
                    [15.0931, -7.9552, 15.6675, -7.0056, 0.5979]],
                   dtype=np.float32)
  scores = np.array([3, 1.1, 5, 2, 1, 0], dtype=np.float32)
  nms_overlap_thresh = 0.7
  keep_indices = ml3d.ops.nms(boxes, scores, nms_overlap_thresh)
  print(keep_indices)

  # PyTorch example.
  import torch
  import open3d.ml.torch as ml3d

  boxes = torch.Tensor([[15.0811, -7.9803, 15.6721, -6.8714, 0.5152],
                        [15.1166, -7.9261, 15.7060, -6.8137, 0.6501],
                        [15.1304, -7.8129, 15.7069, -6.8903, 0.7296],
                        [15.2050, -7.8447, 15.8311, -6.7437, 1.0506],
                        [15.1343, -7.8136, 15.7121, -6.8479, 1.0352],
                        [15.0931, -7.9552, 15.6675, -7.0056, 0.5979]])
  scores = torch.Tensor([3, 1.1, 5, 2, 1, 0])
  nms_overlap_thresh = 0.7
  keep_indices = ml3d.ops.nms(boxes, scores, nms_overlap_thresh)
  print(keep_indices)

boxes: (N, 5) float32 tensor. Bounding boxes are represented as (x0, y0, x1, y1, rotate).

scores: (N,) float32 tensor. A higher score means a more confident bounding box.

nms_overlap_thresh: float value between 0 and 1. When a high-score box is
  selected, other remaining boxes with IoU > nms_overlap_thresh will be discarded.
  A higher nms_overlap_thresh means more boxes will be kept.

returns (M,) int64 tensor. The selected box indices.
)doc");
