// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/geometry/RGBDImage.h"
#include "open3d/pipelines/odometry/RGBDOdometryJacobian.h"
#include "tests/Tests.h"
#include "tests/pipelines/odometry/OdometryTools.h"

namespace open3d {
namespace tests {

using namespace odometry_tools;

TEST(RGBDOdometryJacobianFromColorTerm, ComputeJacobianAndResidual) {
    std::vector<Eigen::Vector6d, utility::Vector6d_allocator> ref_J_r(10);
    ref_J_r[0] << -1.208103, 0.621106, -0.040830, 0.173142, 0.260220, -1.164557;
    ref_J_r[1] << -0.338017, 0.140257, 0.019732, 0.030357, 0.128839, -0.395772;
    ref_J_r[2] << -0.235842, 0.122008, 0.029948, 0.037260, 0.119792, -0.194611;
    ref_J_r[3] << -0.222063, 0.118091, -0.018617, 0.096335, 0.144784, -0.230677;
    ref_J_r[4] << -0.127762, 0.197381, 0.104905, 0.072993, 0.146487, -0.186723;
    ref_J_r[5] << -0.012070, 0.033963, -0.004087, 0.019158, 0.004083, -0.022654;
    ref_J_r[6] << -0.047053, 0.049144, -0.027889, 0.040064, 0.010937, -0.048321;
    ref_J_r[7] << -0.338017, 0.140257, 0.019732, 0.030357, 0.128839, -0.395772;
    ref_J_r[8] << -2.080471, 1.779082, 0.191770, 0.116250, 0.373750, -2.206175;
    ref_J_r[9] << -0.015476, 0.054573, -0.002288, 0.027828, 0.005931, -0.046776;

    std::vector<double> ref_r = {0.419608, -0.360784, 0.274510,  0.564706,
                                 0.835294, -0.352941, -0.545098, -0.360784,
                                 0.121569, -0.094118};

    int width = 10;
    int height = 10;
    // int num_of_channels = 1;
    // int bytes_per_channel = 4;

    auto srcColor = GenerateImage(width, height, 1, 4, 0.0f, 1.0f, 1);
    auto srcDepth = GenerateImage(width, height, 1, 4, 0.0f, 1.0f, 0);

    auto tgtColor = GenerateImage(width, height, 1, 4, 0.0f, 1.0f, 1);
    auto tgtDepth = GenerateImage(width, height, 1, 4, 1.0f, 2.0f, 0);

    auto dxColor = GenerateImage(width, height, 1, 4, 0.0f, 1.0f, 1);
    auto dyColor = GenerateImage(width, height, 1, 4, 0.0f, 1.0f, 1);

    ShiftLeft(tgtColor, 10);
    ShiftUp(tgtColor, 5);

    ShiftLeft(dxColor, 10);
    ShiftUp(dyColor, 5);

    geometry::RGBDImage source(*srcColor, *srcDepth);
    geometry::RGBDImage target(*tgtColor, *tgtDepth);
    auto source_xyz = GenerateImage(width, height, 3, 4, 0.0f, 1.0f, 0);
    geometry::RGBDImage target_dx(*dxColor, *tgtDepth);
    geometry::RGBDImage target_dy(*dyColor, *tgtDepth);

    Eigen::Matrix3d intrinsic = Eigen::Matrix3d::Zero();
    intrinsic(0, 0) = 0.5;
    intrinsic(1, 1) = 0.65;
    intrinsic(0, 2) = 0.75;
    intrinsic(1, 2) = 0.35;

    Eigen::Matrix4d extrinsic = Eigen::Matrix4d::Zero();
    extrinsic(0, 0) = 1.0;
    extrinsic(1, 1) = 1.0;
    extrinsic(2, 2) = 1.0;

    int rows = height;
    std::vector<Eigen::Vector4i, utility::Vector4i_allocator> corresps(rows);
    Rand(corresps, 0, 3, 0);

    pipelines::odometry::RGBDOdometryJacobianFromColorTerm jacobian_method;

    for (int row = 0; row < rows; row++) {
        std::vector<Eigen::Vector6d, utility::Vector6d_allocator> J_r;
        std::vector<double> r;
        std::vector<double> w;

        jacobian_method.ComputeJacobianAndResidual(
                row, J_r, r, w, source, target, *source_xyz, target_dx,
                target_dy, intrinsic, extrinsic, corresps);

        EXPECT_NEAR(ref_r[row], r[0], THRESHOLD_1E_6);
        ExpectEQ(ref_J_r[row], J_r[0], 1e-4);
    }
}

}  // namespace tests
}  // namespace open3d
