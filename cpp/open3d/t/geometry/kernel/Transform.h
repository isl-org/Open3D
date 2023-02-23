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
namespace transform {

void TransformPoints(const core::Tensor& transformation, core::Tensor& points);

void TransformNormals(const core::Tensor& transformation,
                      core::Tensor& normals);

void RotatePoints(const core::Tensor& R,
                  core::Tensor& points,
                  const core::Tensor& center);

void RotateNormals(const core::Tensor& R, core::Tensor& normals);

void TransformPointsCPU(const core::Tensor& transformation,
                        core::Tensor& points);

void TransformNormalsCPU(const core::Tensor& transformation,
                         core::Tensor& normals);

void RotatePointsCPU(const core::Tensor& R,
                     core::Tensor& points,
                     const core::Tensor& center);

void RotateNormalsCPU(const core::Tensor& R, core::Tensor& normals);

#ifdef BUILD_CUDA_MODULE
void TransformPointsCUDA(const core::Tensor& transformation,
                         core::Tensor& points);

void TransformNormalsCUDA(const core::Tensor& transformation,
                          core::Tensor& normals);

void RotatePointsCUDA(const core::Tensor& R,
                      core::Tensor& points,
                      const core::Tensor& center);

void RotateNormalsCUDA(const core::Tensor& R, core::Tensor& normals);
#endif

}  // namespace transform
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
