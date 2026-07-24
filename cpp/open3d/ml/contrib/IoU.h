// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#ifdef BUILD_SYCL_MODULE
#include "open3d/core/Device.h"
#endif

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

#ifdef BUILD_SYCL_MODULE
/// \param device SYCL device; uses SYCLContext's default queue for that device.
/// \param boxes_a (num_a, 5) float32 on \p device.
/// \param boxes_b (num_b, 5) float32 on \p device.
/// \param iou (num_a, num_b) float32 on \p device, output iou values.
/// \param num_a Number of boxes in boxes_a.
/// \param num_b Number of boxes in boxes_b.
void IoUBevSYCLKernel(const core::Device &device,
                      const float *boxes_a,
                      const float *boxes_b,
                      float *iou,
                      int num_a,
                      int num_b);

/// \param device SYCL device; uses SYCLContext's default queue for that device.
/// \param boxes_a (num_a, 7) float32 on \p device.
/// \param boxes_b (num_b, 7) float32 on \p device.
/// \param iou (num_a, num_b) float32 on \p device, output iou values.
/// \param num_a Number of boxes in boxes_a.
/// \param num_b Number of boxes in boxes_b.
void IoU3dSYCLKernel(const core::Device &device,
                     const float *boxes_a,
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
