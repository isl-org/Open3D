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

#include "open3d/core/Tensor.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace image {

void To(const core::Tensor &src,
        core::Tensor &dst,
        double scale,
        double offset);

void ClipTransform(const core::Tensor &src,
                   core::Tensor &dst,
                   float scale,
                   float min_value,
                   float max_value,
                   float clip_fill = 0.0f);

void PyrDownDepth(const core::Tensor &src,
                  core::Tensor &dst,
                  float diff_threshold,
                  float invalid_fill);

void CreateVertexMap(const core::Tensor &src,
                     core::Tensor &dst,
                     const core::Tensor &intrinsics,
                     float invalid_fill);

void CreateNormalMap(const core::Tensor &src,
                     core::Tensor &dst,
                     float invalid_fill);

void ColorizeDepth(const core::Tensor &src,
                   core::Tensor &dst,
                   float scale,
                   float min_value,
                   float max_value);

void ToCPU(const core::Tensor &src,
           core::Tensor &dst,
           double scale,
           double offset);

void ClipTransformCPU(const core::Tensor &src,
                      core::Tensor &dst,
                      float scale,
                      float min_value,
                      float max_value,
                      float clip_fill = 0.0f);

void PyrDownDepthCPU(const core::Tensor &src,
                     core::Tensor &dst,
                     float diff_threshold,
                     float invalid_fill);

void CreateVertexMapCPU(const core::Tensor &src,
                        core::Tensor &dst,
                        const core::Tensor &intrinsics,
                        float invalid_fill);

void CreateNormalMapCPU(const core::Tensor &src,
                        core::Tensor &dst,
                        float invalid_fill);

void ColorizeDepthCPU(const core::Tensor &src,
                      core::Tensor &dst,
                      float scale,
                      float min_value,
                      float max_value);

#ifdef BUILD_CUDA_MODULE
void ToCUDA(const core::Tensor &src,
            core::Tensor &dst,
            double scale,
            double offset);

void ClipTransformCUDA(const core::Tensor &src,
                       core::Tensor &dst,
                       float scale,
                       float min_value,
                       float max_value,
                       float clip_fill = 0.0f);

void PyrDownDepthCUDA(const core::Tensor &src,
                      core::Tensor &dst,
                      float diff_threshold,
                      float invalid_fill);

void CreateVertexMapCUDA(const core::Tensor &src,
                         core::Tensor &dst,
                         const core::Tensor &intrinsics,
                         float invalid_fill);

void CreateNormalMapCUDA(const core::Tensor &src,
                         core::Tensor &dst,
                         float invalid_fill);

void ColorizeDepthCUDA(const core::Tensor &src,
                       core::Tensor &dst,
                       float scale,
                       float min_value,
                       float max_value);

#endif
}  // namespace image
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
