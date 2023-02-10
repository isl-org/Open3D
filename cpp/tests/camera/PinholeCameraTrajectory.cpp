// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/camera/PinholeCameraTrajectory.h"

#include <json/json.h>

#include "tests/Tests.h"

namespace open3d {
namespace tests {

TEST(PinholeCameraTrajectory, DISABLED_MemberData) { NotImplemented(); }

TEST(PinholeCameraTrajectory, ConvertToFromJsonValue) {
    camera::PinholeCameraTrajectory src;
    camera::PinholeCameraTrajectory dst;

    int width = 640;
    int height = 480;

    src.parameters_.resize(2);
    for (size_t i = 0; i < src.parameters_.size(); i++) {
        camera::PinholeCameraIntrinsic intrinsic;
        intrinsic.width_ = width;
        intrinsic.height_ = height;
        intrinsic.intrinsic_matrix_ = Eigen::Matrix3d::Random();

        src.parameters_[i].intrinsic_ = intrinsic;
        src.parameters_[i].extrinsic_ = Eigen::Matrix4d::Random();
    }

    Json::Value value;
    bool output = src.ConvertToJsonValue(value);
    EXPECT_TRUE(output);

    output = dst.ConvertFromJsonValue(value);
    EXPECT_TRUE(output);

    EXPECT_EQ(src.parameters_.size(), dst.parameters_.size());

    for (size_t i = 0; i < src.parameters_.size(); i++) {
        camera::PinholeCameraParameters src_params = src.parameters_[i];
        camera::PinholeCameraParameters dst_params = dst.parameters_[i];

        EXPECT_EQ(src_params.intrinsic_.width_, dst_params.intrinsic_.width_);
        EXPECT_EQ(src_params.intrinsic_.height_, dst_params.intrinsic_.height_);

        ExpectEQ(src_params.intrinsic_.intrinsic_matrix_,
                 dst_params.intrinsic_.intrinsic_matrix_);

        ExpectEQ(src_params.extrinsic_, dst_params.extrinsic_);
    }
}

}  // namespace tests
}  // namespace open3d
