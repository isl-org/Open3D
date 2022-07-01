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

}  // namespace odometry
}  // namespace pipelines
}  // namespace open3d
