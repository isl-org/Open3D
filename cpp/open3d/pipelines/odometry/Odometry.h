// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <iostream>
#include <tuple>
#include <vector>

#include "open3d/camera/PinholeCameraIntrinsic.h"
#include "open3d/pipelines/odometry/OdometryOption.h"
#include "open3d/pipelines/odometry/RGBDOdometryJacobian.h"
#include "open3d/utility/Eigen.h"
#include "open3d/utility/Logging.h"

namespace open3d {

namespace geometry {
class RGBDImage;
}

namespace pipelines {
namespace odometry {

/// \brief Function to estimate 6D rigid motion from two RGBD image pairs.
///
/// \param source Source RGBD image.
/// \param target Target RGBD image.
/// \param pinhole_camera_intrinsic Camera intrinsic parameters.
/// \param odo_init Initial 4x4 motion matrix estimation.
/// \param jacobian_method The odometry Jacobian method to use.
/// \param option Odometry hyper parameters.
/// \return is_success, 4x4 motion matrix, 6x6 information matrix.
std::tuple<bool, Eigen::Matrix4d, Eigen::Matrix6d> ComputeRGBDOdometry(
        const geometry::RGBDImage &source,
        const geometry::RGBDImage &target,
        const camera::PinholeCameraIntrinsic &pinhole_camera_intrinsic =
                camera::PinholeCameraIntrinsic(),
        const Eigen::Matrix4d &odo_init = Eigen::Matrix4d::Identity(),
        const RGBDOdometryJacobian &jacobian_method =
                RGBDOdometryJacobianFromHybridTerm(),
        const OdometryOption &option = OdometryOption());

/// \brief Function to estimate point to point correspondences from two depth
/// images.
///
/// \param intrinsic_matrix Camera intrinsic parameters.
/// \param extrinsic Estimation of transform from source to target.
/// \param depth_s Source depth image.
/// \param depth_t Target depth image.
/// \param option Odometry hyper parameters.
/// \return A vector of u_s, v_s, u_t, v_t which maps the 2d coordinates of
/// source to target.
CorrespondenceSetPixelWise ComputeCorrespondence(
        const Eigen::Matrix3d &intrinsic_matrix,
        const Eigen::Matrix4d &extrinsic,
        const geometry::Image &depth_s,
        const geometry::Image &depth_t,
        const OdometryOption &option);

}  // namespace odometry
}  // namespace pipelines
}  // namespace open3d
