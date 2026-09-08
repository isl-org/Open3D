// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//***************************************************************************************/
//
//    Based on PointRCNN Library (MIT License):
//    https://github.com/sshaoshuai/PointRCNN
//
//    Copyright (c) 2019 Shaoshuai Shi
//
//    Permission is hereby granted, free of charge, to any person obtaining a
//    copy of this software and associated documentation files (the "Software"),
//    to deal in the Software without restriction, including without limitation
//    the rights to use, copy, modify, merge, publish, distribute, sublicense,
//    and/or sell copies of the Software, and to permit persons to whom the
//    Software is furnished to do so, subject to the following conditions:
//
//    The above copyright notice and this permission notice shall be included in
//    all copies or substantial portions of the Software.
//
//    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//    DEALINGS IN THE SOFTWARE.
//
//***************************************************************************************/

#pragma once

#include <math.h>

#include "open3d/Macro.h"
#include "open3d/core/CUDAUtils.h"

#ifdef BUILD_SYCL_MODULE
// roipool3dLauncherSYCL takes a real sycl::queue&, so any TU that sees this
// declaration needs the full SYCL runtime type (not -fsycl compilation --
// RoiPoolOps.cpp already gets this transitively via
// <c10/xpu/XPUDeviceProp.h>, but this header may be included first, so it
// must include the SYCL header directly to avoid an incomplete-type error).
// RoiPoolKernelSYCL.cpp (built with -fsycl) is where the actual device
// kernels are compiled.
#include <sycl/sycl.hpp>
#endif

namespace open3d {
namespace ml {
namespace contrib {

/// Checks whether point (x, y, z) lies inside the rotated 3D box centered
/// at (cx, cy=bottom_y-h/2, cz) with size (h, w, l) and heading \p angle.
/// Shared by the CPU/CUDA/SYCL roi_pool launchers. Returns bool (not the
/// original PointRCNN int) since callers only ever store it into an int
/// flag array or test it for truthiness; bool->int conversion is well
/// defined (true/false -> 1/0) so all three backends are unaffected.
OPEN3D_HOST_DEVICE inline bool pt_in_box3d(float x,
                                           float y,
                                           float z,
                                           float cx,
                                           float bottom_y,
                                           float cz,
                                           float h,
                                           float w,
                                           float l,
                                           float angle,
                                           float max_dis) {
    float x_rot, z_rot, cosa, sina, cy;
    cy = bottom_y - h / 2.0;
    if ((fabsf(x - cx) > max_dis) || (fabsf(y - cy) > h / 2.0) ||
        (fabsf(z - cz) > max_dis)) {
        return false;
    }
    cosa = cos(angle);
    sina = sin(angle);
    x_rot = (x - cx) * cosa + (z - cz) * (-sina);
    z_rot = (x - cx) * sina + (z - cz) * cosa;

    return (x_rot >= -l / 2.0) & (x_rot <= l / 2.0) & (z_rot >= -w / 2.0) &
           (z_rot <= w / 2.0);
}

#ifdef BUILD_CUDA_MODULE

void roipool3dLauncher(int batch_size,
                       int pts_num,
                       int boxes_num,
                       int feature_in_len,
                       int sampled_pts_num,
                       const float *xyz,
                       const float *boxes3d,
                       const float *pts_feature,
                       float *pooled_features,
                       int *pooled_empty_flag);
#endif

#ifdef BUILD_SYCL_MODULE

void roipool3dLauncherSYCL(sycl::queue &queue,
                           int batch_size,
                           int pts_num,
                           int boxes_num,
                           int feature_in_len,
                           int sampled_pts_num,
                           const float *xyz,
                           const float *boxes3d,
                           const float *pts_feature,
                           float *pooled_features,
                           int *pooled_empty_flag);
#endif

void roipool3dLauncherCPU(int batch_size,
                          int pts_num,
                          int boxes_num,
                          int feature_in_len,
                          int sampled_pts_num,
                          const float *xyz,
                          const float *boxes3d,
                          const float *pts_feature,
                          float *pooled_features,
                          int *pooled_empty_flag);

}  // namespace contrib
}  // namespace ml
}  // namespace open3d
