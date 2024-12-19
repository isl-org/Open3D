// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/camera/PinholeCameraIntrinsic.h"

#include <json/json.h>

#include "tests/Tests.h"

namespace open3d {
namespace tests {

TEST(PinholeCameraIntrinsic, Constructor_Default) {
    camera::PinholeCameraIntrinsic intrinsic;

    EXPECT_EQ(-1, intrinsic.width_);
    EXPECT_EQ(-1, intrinsic.height_);

    Eigen::Matrix3d reference = Eigen::Matrix3d::Identity();
    ExpectEQ(reference, intrinsic.intrinsic_matrix_);
}

TEST(PinholeCameraIntrinsic, Constructor_PrimeSenseDefault) {
    camera::PinholeCameraIntrinsic intrinsic(
            camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);

    EXPECT_EQ(640, intrinsic.width_);
    EXPECT_EQ(480, intrinsic.height_);

    Eigen::Matrix3d reference;
    reference << 525.0, 0.0, 319.5, 0.0, 525.0, 239.5, 0.0, 0.0, 1.0;

    ExpectEQ(reference, intrinsic.intrinsic_matrix_);
}

TEST(PinholeCameraIntrinsic, Constructor_Kinect2DepthCameraDefault) {
    camera::PinholeCameraIntrinsic intrinsic(
            camera::PinholeCameraIntrinsicParameters::
                    Kinect2DepthCameraDefault);

    EXPECT_EQ(512, intrinsic.width_);
    EXPECT_EQ(424, intrinsic.height_);

    Eigen::Matrix3d reference;
    reference << 365.456, 0.0, 254.878, 0.0, 365.456, 205.395, 0.0, 0.0, 1.0;

    ExpectEQ(reference, intrinsic.intrinsic_matrix_);
}

TEST(PinholeCameraIntrinsic, Constructor_Kinect2ColorCameraDefault) {
    camera::PinholeCameraIntrinsic intrinsic(
            camera::PinholeCameraIntrinsicParameters::
                    Kinect2ColorCameraDefault);

    EXPECT_EQ(1920, intrinsic.width_);
    EXPECT_EQ(1080, intrinsic.height_);

    Eigen::Matrix3d reference;
    reference << 1059.9718, 0.0, 975.7193, 0.0, 1059.9718, 545.9533, 0.0, 0.0,
            1.0;

    ExpectEQ(reference, intrinsic.intrinsic_matrix_);
}

TEST(PinholeCameraIntrinsic, Constructor_Init) {
    int width = 640;
    int height = 480;

    double fx = 0.5;
    double fy = 0.65;

    double cx = 0.75;
    double cy = 0.35;

    camera::PinholeCameraIntrinsic intrinsic(width, height, fx, fy, cx, cy);

    EXPECT_EQ(width, intrinsic.width_);
    EXPECT_EQ(height, intrinsic.height_);

    Eigen::Matrix3d reference;
    reference << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;

    ExpectEQ(reference, intrinsic.intrinsic_matrix_);
}

TEST(PinholeCameraIntrinsic, DISABLED_MemberData) { NotImplemented(); }

TEST(PinholeCameraIntrinsic, SetIntrinsics) {
    camera::PinholeCameraIntrinsic intrinsic;

    EXPECT_EQ(-1, intrinsic.width_);
    EXPECT_EQ(-1, intrinsic.height_);

    intrinsic.intrinsic_matrix_ = Eigen::Matrix3d::Zero();

    int width = 640;
    int height = 480;

    double fx = 0.5;
    double fy = 0.65;

    double cx = 0.75;
    double cy = 0.35;

    intrinsic.SetIntrinsics(width, height, fx, fy, cx, cy);

    EXPECT_EQ(width, intrinsic.width_);
    EXPECT_EQ(height, intrinsic.height_);

    Eigen::Matrix3d reference;
    reference << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;

    ExpectEQ(reference, intrinsic.intrinsic_matrix_);
}

TEST(PinholeCameraIntrinsic, GetFocalLength) {
    camera::PinholeCameraIntrinsic intrinsic;

    int width = 640;
    int height = 480;

    double fx = 0.5;
    double fy = 0.65;

    double cx = 0.75;
    double cy = 0.35;

    intrinsic.SetIntrinsics(width, height, fx, fy, cx, cy);

    EXPECT_EQ(width, intrinsic.width_);
    EXPECT_EQ(height, intrinsic.height_);

    std::pair<double, double> output = intrinsic.GetFocalLength();

    EXPECT_NEAR(fx, output.first, THRESHOLD_1E_6);
    EXPECT_NEAR(fy, output.second, THRESHOLD_1E_6);
}

TEST(PinholeCameraIntrinsic, GetPrincipalPoint) {
    camera::PinholeCameraIntrinsic intrinsic;

    int width = 640;
    int height = 480;

    double fx = 0.5;
    double fy = 0.65;

    double cx = 0.75;
    double cy = 0.35;

    intrinsic.SetIntrinsics(width, height, fx, fy, cx, cy);

    EXPECT_EQ(width, intrinsic.width_);
    EXPECT_EQ(height, intrinsic.height_);

    std::pair<double, double> output = intrinsic.GetPrincipalPoint();

    EXPECT_NEAR(cx, output.first, THRESHOLD_1E_6);
    EXPECT_NEAR(cy, output.second, THRESHOLD_1E_6);
}

TEST(PinholeCameraIntrinsic, GetSkew) {
    camera::PinholeCameraIntrinsic intrinsic;

    int width = 640;
    int height = 480;

    double fx = 0.5;
    double fy = 0.65;

    double cx = 0.75;
    double cy = 0.35;

    intrinsic.SetIntrinsics(width, height, fx, fy, cx, cy);

    EXPECT_EQ(width, intrinsic.width_);
    EXPECT_EQ(height, intrinsic.height_);

    double output = intrinsic.GetSkew();

    EXPECT_NEAR(0.0, output, THRESHOLD_1E_6);
}

TEST(PinholeCameraIntrinsic, IsValid) {
    camera::PinholeCameraIntrinsic intrinsic;

    EXPECT_FALSE(intrinsic.IsValid());

    int width = 640;
    int height = 480;

    double fx = 0.5;
    double fy = 0.65;

    double cx = 0.75;
    double cy = 0.35;

    intrinsic.SetIntrinsics(width, height, fx, fy, cx, cy);

    EXPECT_TRUE(intrinsic.IsValid());
}

TEST(PinholeCameraIntrinsic, ConvertToFromJsonValue) {
    camera::PinholeCameraIntrinsic src;
    camera::PinholeCameraIntrinsic dst;

    int width = 640;
    int height = 480;

    double fx = 0.5;
    double fy = 0.65;

    double cx = 0.75;
    double cy = 0.35;

    src.SetIntrinsics(width, height, fx, fy, cx, cy);

    Json::Value value;
    bool output = src.ConvertToJsonValue(value);
    EXPECT_TRUE(output);

    output = dst.ConvertFromJsonValue(value);
    EXPECT_TRUE(output);

    EXPECT_EQ(width, dst.width_);
    EXPECT_EQ(height, dst.height_);

    Eigen::Matrix3d reference;
    reference << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;

    ExpectEQ(reference, dst.intrinsic_matrix_);
}

}  // namespace tests
}  // namespace open3d
