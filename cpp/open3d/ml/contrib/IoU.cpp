// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/ml/contrib/IoU.h"

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
    tbb::parallel_for(0, num_a, [&](int idx_a) {
        tbb::parallel_for(0, num_b, [&](int idx_b) {
            const float *box_a = boxes_a + idx_a * 5;
            const float *box_b = boxes_b + idx_b * 5;
            float *out = iou + idx_a * num_b + idx_b;
            *out = IoUBev2DWithCenterAndSize(box_a, box_b);
        });
    });
}

void IoU3dCPUKernel(const float *boxes_a,
                    const float *boxes_b,
                    float *iou,
                    int num_a,
                    int num_b) {
    tbb::parallel_for(0, num_a, [&](int idx_a) {
        tbb::parallel_for(0, num_b, [&](int idx_b) {
            const float *box_a = boxes_a + idx_a * 7;
            const float *box_b = boxes_b + idx_b * 7;
            float *out = iou + idx_a * num_b + idx_b;
            *out = IoU3DWithCenterAndSize(box_a, box_b);
        });
    });
}

}  // namespace contrib
}  // namespace ml
}  // namespace open3d
