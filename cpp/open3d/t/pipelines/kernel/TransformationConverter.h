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
namespace pipelines {
namespace kernel {

/// \brief Convert rotation and translation to the transformation matrix.
///
/// \param R Rotation, a tensor of shape {3, 3}.
/// \param t Translation, a tensor of shape {3,}.
/// \return Transformation, a tensor of shape {4, 4}, dtype and device same as R
/// and t.
core::Tensor RtToTransformation(const core::Tensor &R, const core::Tensor &t);

/// \brief Convert pose to the transformation matrix.
///
/// \param pose Pose [alpha beta gamma, tx, ty, tz], a shape {6} tensor of dtype
/// Float32, where alpha, beta, gamma are the Euler angles in the ZYX order.
/// \return Transformation, a tensor of shape {4, 4}, dtype and device same
/// as pose.
core::Tensor PoseToTransformation(const core::Tensor &pose);

/// \brief Decodes a 6x6 linear system from a compressed 29x1 tensor.
/// \param A_reduction 1x29 tensor storing a linear system,
/// (21 for \f$J^T J\f$ matrix, 6 for \f$J^T r\f$, 1 for residual,
/// 1 for inlier count).
/// \param delta 6d tensor for a se3 tangent vector.
/// \param inlier_residual Float residual for the inliers.
/// \param inlier_count Int number of inliers.
void DecodeAndSolve6x6(const core::Tensor &A_reduction,
                       core::Tensor &delta,
                       float &inlier_residual,
                       int &inlier_count);

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
