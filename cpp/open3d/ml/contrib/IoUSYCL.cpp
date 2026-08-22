// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// SYCL implementation of IoU — ports IoU.cu. Each (box_a, box_b) pair is
// independent, so unlike the CUDA grid-stride-loop kernel this uses a flat
// 1-D nd_range over num_a*num_b elements via core::ParallelFor instead of a
// bare sycl::range, for better occupancy control.

#include "open3d/core/ParallelFor.h"
#include "open3d/core/SYCLContext.h"
#include "open3d/ml/contrib/IoU.h"
#include "open3d/ml/contrib/IoUImpl.h"

namespace open3d {
namespace ml {
namespace contrib {

void IoUBevSYCLKernel(sycl::queue &queue,
                      const float *boxes_a,
                      const float *boxes_b,
                      float *iou,
                      int num_a,
                      int num_b) {
    const int n = num_a * num_b;
    if (n == 0) {
        return;
    }
    core::ParallelFor(queue, n, [=](int64_t idx64) {
        const int idx = static_cast<int>(idx64);
        const int idx_a = idx / num_b;
        const int idx_b = idx % num_b;
        const float *box_a = boxes_a + idx_a * 5;
        const float *box_b = boxes_b + idx_b * 5;
        iou[idx_a * num_b + idx_b] = IoUBev2DWithCenterAndSize(box_a, box_b);
    });
}

void IoU3dSYCLKernel(sycl::queue &queue,
                     const float *boxes_a,
                     const float *boxes_b,
                     float *iou,
                     int num_a,
                     int num_b) {
    const int n = num_a * num_b;
    if (n == 0) {
        return;
    }
    core::ParallelFor(queue, n, [=](int64_t idx64) {
        const int idx = static_cast<int>(idx64);
        const int idx_a = idx / num_b;
        const int idx_b = idx % num_b;
        const float *box_a = boxes_a + idx_a * 7;
        const float *box_b = boxes_b + idx_b * 7;
        iou[idx_a * num_b + idx_b] = IoU3DWithCenterAndSize(box_a, box_b);
    });
}

// core::Device overloads: resolve to SYCLContext's default queue here (this
// TU IS -fsycl-compiled) and forward, so non-SYCL-compiled callers (the
// pybind bindings in iou.cpp) never need to spell sycl::queue themselves.
void IoUBevSYCLKernel(const core::Device &device,
                      const float *boxes_a,
                      const float *boxes_b,
                      float *iou,
                      int num_a,
                      int num_b) {
    sycl::queue queue = core::sy::GetQueue(device);
    IoUBevSYCLKernel(queue, boxes_a, boxes_b, iou, num_a, num_b);
}

void IoU3dSYCLKernel(const core::Device &device,
                     const float *boxes_a,
                     const float *boxes_b,
                     float *iou,
                     int num_a,
                     int num_b) {
    sycl::queue queue = core::sy::GetQueue(device);
    IoU3dSYCLKernel(queue, boxes_a, boxes_b, iou, num_a, num_b);
}

}  // namespace contrib
}  // namespace ml
}  // namespace open3d
