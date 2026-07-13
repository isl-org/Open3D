// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// XPU dispatch wrapper for TrilinearDevoxelize(Grad): obtains the current
// SYCL queue from PyTorch XPU context and delegates to the SYCL impl in
// TrilinearDevoxelizeSYCL.h.

#include <c10/xpu/XPUStream.h>

#include "open3d/ml/impl/contrib/TrilinearDevoxelizeSYCL.h"
#include "open3d/ml/pytorch/pvcnn/TrilinearDevoxelizeKernel.h"

void TrilinearDevoxelizeSYCLLauncher(int b,
                                     int c,
                                     int n,
                                     int r,
                                     int r2,
                                     int r3,
                                     bool is_training,
                                     const float *coords,
                                     const float *feat,
                                     int *inds,
                                     float *wgts,
                                     float *outs) {
    sycl::queue &queue = c10::xpu::getCurrentXPUStream().queue();
    open3d::ml::contrib::TrilinearDevoxelizeSYCL(queue, b, c, n, r, r2, r3,
                                                 is_training, coords, feat,
                                                 inds, wgts, outs);
}

void TrilinearDevoxelizeGradSYCLLauncher(int b,
                                         int c,
                                         int n,
                                         int r3,
                                         const int *inds,
                                         const float *wgts,
                                         const float *grad_y,
                                         float *grad_x) {
    sycl::queue &queue = c10::xpu::getCurrentXPUStream().queue();
    open3d::ml::contrib::TrilinearDevoxelizeGradSYCL(queue, b, c, n, r3, inds,
                                                     wgts, grad_y, grad_x);
}
