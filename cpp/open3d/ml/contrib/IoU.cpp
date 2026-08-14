// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/ml/contrib/IoU.h"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include "open3d/ml/contrib/IoUImpl.h"

namespace open3d {
namespace ml {
namespace contrib {

void IoUBevCPUKernel(const float *boxes_a,
                     const float *boxes_b,
                     float *iou,
                     int num_a,
                     int num_b) {
    // Use a single flattened parallel_for to avoid nested parallelism.
    // Each (idx_a, idx_b) pair is independent — no data races.
    const int total = num_a * num_b;
    tbb::parallel_for(tbb::blocked_range<int>(0, total),
                      [&](const tbb::blocked_range<int> &r) {
                          for (int flat_idx = r.begin(); flat_idx != r.end();
                               ++flat_idx) {
                              const int idx_a = flat_idx / num_b;
                              const int idx_b = flat_idx % num_b;
                              const float *box_a = boxes_a + idx_a * 5;
                              const float *box_b = boxes_b + idx_b * 5;
                              float *out = iou + flat_idx;
                              *out = IoUBev2DWithCenterAndSize(box_a, box_b);
                          }
                      });
}

void IoU3dCPUKernel(const float *boxes_a,
                    const float *boxes_b,
                    float *iou,
                    int num_a,
                    int num_b) {
    // Use a single flattened parallel_for to avoid nested parallelism.
    // Each (idx_a, idx_b) pair is independent — no data races.
    const int total = num_a * num_b;
    tbb::parallel_for(tbb::blocked_range<int>(0, total),
                      [&](const tbb::blocked_range<int> &r) {
                          for (int flat_idx = r.begin(); flat_idx != r.end();
                               ++flat_idx) {
                              const int idx_a = flat_idx / num_b;
                              const int idx_b = flat_idx % num_b;
                              const float *box_a = boxes_a + idx_a * 7;
                              const float *box_b = boxes_b + idx_b * 7;
                              float *out = iou + flat_idx;
                              *out = IoU3DWithCenterAndSize(box_a, box_b);
                          }
                      });
}

}  // namespace contrib
}  // namespace ml
}  // namespace open3d
