// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
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

void ComputeTriangleAreasCPU(const core::Tensor& vertices,
                             const core::Tensor& triangles,
                             core::Tensor& triangle_areas);

std::array<core::Tensor, 3> SamplePointsUniformlyCPU(
        const core::Tensor& triangles,
        const core::Tensor& vertices,
        const core::Tensor& triangle_areas,
        const core::Tensor& vertex_normals,
        const core::Tensor& vertex_colors,
        const core::Tensor& triangle_normals,
        const core::Tensor& texture_uvs,
        const core::Tensor& albedo,
        size_t number_of_points);

#ifdef BUILD_CUDA_MODULE
void NormalizeNormalsCUDA(core::Tensor& normals);

void ComputeTriangleNormalsCUDA(const core::Tensor& vertices,
                                const core::Tensor& triangles,
                                core::Tensor& normals);

void ComputeVertexNormalsCUDA(const core::Tensor& triangles,
                              const core::Tensor& triangle_normals,
                              core::Tensor& vertex_normals);

void ComputeTriangleAreasCUDA(const core::Tensor& vertices,
                              const core::Tensor& triangles,
                              core::Tensor& triangle_areas);
#endif

}  // namespace trianglemesh
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
