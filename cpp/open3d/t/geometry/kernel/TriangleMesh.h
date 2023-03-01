// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
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
