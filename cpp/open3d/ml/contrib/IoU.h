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

#pragma once

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
