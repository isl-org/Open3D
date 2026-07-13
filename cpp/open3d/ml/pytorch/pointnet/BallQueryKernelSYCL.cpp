// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// XPU dispatch wrapper for BallQuery: obtains the current SYCL queue from
// PyTorch XPU context and delegates to the SYCL impl in BallQuerySYCL.h.

#include <c10/xpu/XPUStream.h>

#include "open3d/ml/impl/contrib/BallQuerySYCL.h"
#include "open3d/ml/pytorch/pointnet/BallQueryKernel.h"

void ball_query_launcher_sycl(int b,
                              int n,
                              int m,
                              float radius,
                              int nsample,
                              const float *new_xyz,
                              const float *xyz,
                              int *idx) {
    sycl::queue &queue = c10::xpu::getCurrentXPUStream().queue();
    open3d::ml::contrib::BallQuerySYCL(queue, b, n, m, radius, nsample, new_xyz,
                                       xyz, idx);
}
