// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
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
namespace contrib {

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

}  // namespace contrib
}  // namespace ml
}  // namespace open3d
