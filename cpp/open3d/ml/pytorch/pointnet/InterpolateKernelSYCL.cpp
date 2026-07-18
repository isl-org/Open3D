// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// XPU dispatch wrappers for three_nn / three_interpolate(_grad): obtain the
// current SYCL queue from PyTorch XPU context and delegate to the SYCL impl
// in InterpolatePointsSYCL.h.

#include <c10/xpu/XPUStream.h>

#include "open3d/ml/contrib/InterpolatePointsSYCL.h"
#include "open3d/ml/pytorch/pointnet/InterpolateKernel.h"

void three_nn_launcher_sycl(int b,
                            int n,
                            int m,
                            const float *unknown,
                            const float *known,
                            float *dist2,
                            int *idx) {
    sycl::queue &queue = c10::xpu::getCurrentXPUStream().queue();
    open3d::ml::contrib::ThreeNNSYCL(queue, b, n, m, unknown, known, dist2,
                                     idx);
}

void three_interpolate_launcher_sycl(int b,
                                     int c,
                                     int m,
                                     int n,
                                     const float *points,
                                     const int *idx,
                                     const float *weight,
                                     float *out) {
    sycl::queue &queue = c10::xpu::getCurrentXPUStream().queue();
    open3d::ml::contrib::ThreeInterpolateSYCL(queue, b, c, m, n, points, idx,
                                              weight, out);
}

void three_interpolate_grad_launcher_sycl(int b,
                                          int c,
                                          int n,
                                          int m,
                                          const float *grad_out,
                                          const int *idx,
                                          const float *weight,
                                          float *grad_points) {
    sycl::queue &queue = c10::xpu::getCurrentXPUStream().queue();
    open3d::ml::contrib::ThreeInterpolateGradSYCL(queue, b, c, n, m, grad_out,
                                                  idx, weight, grad_points);
}
