// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

namespace open3d {
namespace ml {
namespace contrib {

#ifdef BUILD_CUDA_MODULE
/// \param boxes_a (num_a, 5) float32.
/// \param boxes_b (num_b, 5) float32.
/// \param iou (num_a, num_b) float32, output iou values.
/// \param num_a Number of boxes in boxes_a.
/// \param num_b Number of boxes in boxes_b.
/// intersection over union.
void IoUBevCUDAKernel(const float *boxes_a,
                      const float *boxes_b,
                      float *iou,
                      int num_a,
                      int num_b);

/// \param boxes_a (num_a, 7) float32.
/// \param boxes_b (num_b, 7) float32.
/// \param iou (num_a, num_b) float32, output iou values.
/// \param num_a Number of boxes in boxes_a.
/// \param num_b Number of boxes in boxes_b.
void IoU3dCUDAKernel(const float *boxes_a,
                     const float *boxes_b,
                     float *iou,
                     int num_a,
                     int num_b);

#endif

/// \param boxes_a (num_a, 5) float32.
/// \param boxes_b (num_b, 5) float32.
/// \param iou (num_a, num_b) float32, output iou values.
/// \param num_a Number of boxes in boxes_a.
/// \param num_b Number of boxes in boxes_b.
/// intersection over union.
void IoUBevCPUKernel(const float *boxes_a,
                     const float *boxes_b,
                     float *iou,
                     int num_a,
                     int num_b);

/// \param boxes_a (num_a, 7) float32.
/// \param boxes_b (num_b, 7) float32.
/// \param iou (num_a, num_b) float32, output iou values.
/// \param num_a Number of boxes in boxes_a.
/// \param num_b Number of boxes in boxes_b.
void IoU3dCPUKernel(const float *boxes_a,
                    const float *boxes_b,
                    float *iou,
                    int num_a,
                    int num_b);

}  // namespace contrib
}  // namespace ml
}  // namespace open3d
