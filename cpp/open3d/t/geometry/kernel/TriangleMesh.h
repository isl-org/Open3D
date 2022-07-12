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
namespace trianglemesh {

void NormalizeNormalsCPU(core::Tensor& normals);

void ComputeTriangleNormalsCPU(const core::Tensor& vertices,
                               const core::Tensor& triangles,
                               core::Tensor& normals);

void ComputeVertexNormalsCPU(const core::Tensor& triangles,
                             const core::Tensor& triangle_normals,
                             core::Tensor& vertex_normals);

#ifdef BUILD_CUDA_MODULE
void NormalizeNormalsCUDA(core::Tensor& normals);

void ComputeTriangleNormalsCUDA(const core::Tensor& vertices,
                                const core::Tensor& triangles,
                                core::Tensor& normals);

void ComputeVertexNormalsCUDA(const core::Tensor& triangles,
                              const core::Tensor& triangle_normals,
                              core::Tensor& vertex_normals);
#endif

}  // namespace trianglemesh
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
