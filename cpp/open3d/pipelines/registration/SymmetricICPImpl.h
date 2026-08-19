// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Geometry>
#include <cmath>

#include "open3d/utility/Eigen.h"

namespace open3d {
namespace pipelines {
namespace registration {

inline Eigen::Matrix4d TransformSymmetricPoseToMatrix4d(
        const Eigen::Vector6d &pose,
        const Eigen::Vector3d &source_mean,
        const Eigen::Vector3d &target_mean) {
    const Eigen::Vector3d g = pose.head<3>();
    const double g_norm = g.norm();
    const double theta = std::atan(g_norm);

    Eigen::Matrix3d half_rotation = Eigen::Matrix3d::Identity();
    if (g_norm > 0.0) {
        half_rotation = Eigen::AngleAxisd(theta, g / g_norm).toRotationMatrix();
    }

    // Symmetric ICP solves for a half-angle pose about correspondence means.
    // Equation 11 converts it to the full rigid transformation.
    const Eigen::Matrix3d rotation = half_rotation * half_rotation;
    const Eigen::Vector3d translation =
            target_mean + half_rotation * (pose.tail<3>() * std::cos(theta)) -
            rotation * source_mean;

    Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
    transformation.block<3, 3>(0, 0) = rotation;
    transformation.block<3, 1>(0, 3) = translation;
    return transformation;
}

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
