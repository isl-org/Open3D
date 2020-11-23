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
//
// Reference:
// https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/ops/iou3d_nms/src/iou3d_nms_kernel.cu
//
// Reference:
// https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/ops/iou3d/src/iou3d_kernel.cu
// 3D IoU Calculation and Rotated NMS(modified from 2D NMS written by others)
// Written by Shaoshuai Shi
// All Rights Reserved 2019-2020.

#pragma once

#include <cstdint>
#include <vector>

namespace open3d {
namespace ml {
namespace impl {

#ifdef BUILD_CUDA_MODULE

/// \param boxes (n, 5) float32.
/// \param scores (n,) float32.
/// \param n Number of boxes.
/// \param nms_overlap_thresh When a high-score box is selected, other remaining
/// boxes with IoU > nms_overlap_thresh will be discarded.
/// \return Selected box indices to keep.
std::vector<int64_t> NmsCUDAKernel(const float *boxes,
                                   const float *scores,
                                   int n,
                                   double nms_overlap_thresh);
#endif

/// \param boxes (n, 5) float32.
/// \param scores (n,) float32.
/// \param n Number of boxes.
/// \param nms_overlap_thresh When a high-score box is selected, other remaining
/// boxes with IoU > nms_overlap_thresh will be discarded.
/// \return Selected box indices to keep.
std::vector<int64_t> NmsCPUKernel(const float *boxes,
                                  const float *scores,
                                  int n,
                                  double nms_overlap_thresh);

}  // namespace impl
}  // namespace ml
}  // namespace open3d
