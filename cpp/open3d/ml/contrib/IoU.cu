// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "open3d/ml/contrib/IoU.h"
#include "open3d/ml/contrib/IoUImpl.h"

namespace open3d {
namespace ml {
namespace contrib {

static constexpr int block_size = 128;
static constexpr int thread_size = 4;

__global__ void IoUBevElementKernel(const float *boxes_a,
                                    const float *boxes_b,
                                    float *iou,
                                    int num_a,
                                    int num_b) {
    // Using the "grid-stride loops" pattern.
    int n = num_a * num_b;
    int items_per_block = block_size * thread_size;
    int idx = blockIdx.x * items_per_block + threadIdx.x;
#pragma unroll
    for (int i = 0; i < thread_size; i++) {
        if (idx < n) {
            int idx_a = idx / num_b;
            int idx_b = idx % num_b;

            const float *box_a = boxes_a + idx_a * 5;
            const float *box_b = boxes_b + idx_b * 5;
            float *out = iou + idx_a * num_b + idx_b;
            *out = IoUBev2DWithCenterAndSize(box_a, box_b);

            idx += block_size;
        }
    }
}

void IoUBevCUDAKernel(const float *boxes_a,
                      const float *boxes_b,
                      float *iou,
                      int num_a,
                      int num_b) {
    int n = num_a * num_b;
    if (n == 0) {
        return;
    }
    int items_per_block = block_size * thread_size;
    int grid_size = (n + items_per_block - 1) / items_per_block;
    IoUBevElementKernel<<<grid_size, block_size>>>(boxes_a, boxes_b, iou, num_a,
                                                   num_b);
}

__global__ void IoU3dElementKernel(const float *boxes_a,
                                   const float *boxes_b,
                                   float *iou,
                                   int num_a,
                                   int num_b) {
    // Using the "grid-stride loops" pattern.
    int n = num_a * num_b;
    int items_per_block = block_size * thread_size;
    int idx = blockIdx.x * items_per_block + threadIdx.x;
#pragma unroll
    for (int i = 0; i < thread_size; i++) {
        if (idx < n) {
            int idx_a = idx / num_b;
            int idx_b = idx % num_b;

            const float *box_a = boxes_a + idx_a * 7;
            const float *box_b = boxes_b + idx_b * 7;
            float *out = iou + idx_a * num_b + idx_b;
            *out = IoU3DWithCenterAndSize(box_a, box_b);

            idx += block_size;
        }
    }
}

void IoU3dCUDAKernel(const float *boxes_a,
                     const float *boxes_b,
                     float *iou,
                     int num_a,
                     int num_b) {
    int n = num_a * num_b;
    if (n == 0) {
        return;
    }
    int items_per_block = block_size * thread_size;
    int grid_size = (n + items_per_block - 1) / items_per_block;
    IoU3dElementKernel<<<grid_size, block_size>>>(boxes_a, boxes_b, iou, num_a,
                                                  num_b);
}

}  // namespace contrib
}  // namespace ml
}  // namespace open3d
