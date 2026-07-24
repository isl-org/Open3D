// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// SYCL implementation of IoU — ports IoU.cu. Each (box_a, box_b) pair is
// independent, so unlike the CUDA grid-stride-loop kernel this uses a plain
// 1-D parallel_for over num_a*num_b elements.

#include "open3d/core/SYCLContext.h"
#include "open3d/ml/contrib/IoU.h"
#include "open3d/ml/contrib/IoUImpl.h"

namespace open3d {
namespace ml {
namespace contrib {

namespace {

void IoUBevSYCLKernelImpl(sycl::queue &queue,
                          const float *boxes_a,
                          const float *boxes_b,
                          float *iou,
                          int num_a,
                          int num_b) {
    const int n = num_a * num_b;
    if (n == 0) {
        return;
    }
    queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(n)),
                         [=](sycl::item<1> item) {
                             const int idx = static_cast<int>(item.get_id(0));
                             const int idx_a = idx / num_b;
                             const int idx_b = idx % num_b;
                             const float *box_a = boxes_a + idx_a * 5;
                             const float *box_b = boxes_b + idx_b * 5;
                             iou[idx_a * num_b + idx_b] =
                                     IoUBev2DWithCenterAndSize(box_a, box_b);
                         });
    });
    queue.wait_and_throw();
}

void IoU3dSYCLKernelImpl(sycl::queue &queue,
                         const float *boxes_a,
                         const float *boxes_b,
                         float *iou,
                         int num_a,
                         int num_b) {
    const int n = num_a * num_b;
    if (n == 0) {
        return;
    }
    queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(n)),
                         [=](sycl::item<1> item) {
                             const int idx = static_cast<int>(item.get_id(0));
                             const int idx_a = idx / num_b;
                             const int idx_b = idx % num_b;
                             const float *box_a = boxes_a + idx_a * 7;
                             const float *box_b = boxes_b + idx_b * 7;
                             iou[idx_a * num_b + idx_b] =
                                     IoU3DWithCenterAndSize(box_a, box_b);
                         });
    });
    queue.wait_and_throw();
}

}  // namespace

void IoUBevSYCLKernel(const core::Device &device,
                      const float *boxes_a,
                      const float *boxes_b,
                      float *iou,
                      int num_a,
                      int num_b) {
    sycl::queue queue =
            core::sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    IoUBevSYCLKernelImpl(queue, boxes_a, boxes_b, iou, num_a, num_b);
}

void IoU3dSYCLKernel(const core::Device &device,
                     const float *boxes_a,
                     const float *boxes_b,
                     float *iou,
                     int num_a,
                     int num_b) {
    sycl::queue queue =
            core::sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    IoU3dSYCLKernelImpl(queue, boxes_a, boxes_b, iou, num_a, num_b);
}

}  // namespace contrib
}  // namespace ml
}  // namespace open3d
