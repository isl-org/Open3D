// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#include "UnitTest.h"

#include "Core/Camera/PinholeCameraTrajectory.h"
#include <json/json.h>

using namespace open3d;
using namespace std;
using namespace unit_test;

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(PinholeCameraTrajectory, DISABLED_MemberData)
{
    unit_test::NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(PinholeCameraTrajectory, ConvertToFromJsonValue)
{
    open3d::PinholeCameraTrajectory src;
    open3d::PinholeCameraTrajectory dst;

    int width = 640;
    int height = 480;

    double fx = 0.5;
    double fy = 0.65;

    double cx = 0.75;
    double cy = 0.35;

    src.intrinsic_.SetIntrinsics(width, height, fx, fy, cx, cy);
    src.extrinsic_.push_back(Eigen::Matrix4d::Zero());
    src.extrinsic_[0](0, 0) = 0.0;
    src.extrinsic_[0](1, 0) = 1.0;
    src.extrinsic_[0](2, 0) = 2.0;
    src.extrinsic_[0](3, 0) = 3.0;
    src.extrinsic_[0](0, 1) = 4.0;
    src.extrinsic_[0](1, 1) = 5.0;
    src.extrinsic_[0](2, 1) = 6.0;
    src.extrinsic_[0](3, 1) = 7.0;
    src.extrinsic_[0](0, 2) = 8.0;
    src.extrinsic_[0](1, 2) = 9.0;
    src.extrinsic_[0](2, 2) = 10.0;
    src.extrinsic_[0](3, 2) = 11.0;
    src.extrinsic_[0](0, 3) = 12.0;
    src.extrinsic_[0](1, 3) = 13.0;
    src.extrinsic_[0](2, 3) = 14.0;
    src.extrinsic_[0](3, 3) = 15.0;

    Json::Value value;
    bool output = src.ConvertToJsonValue(value);

    output = dst.ConvertFromJsonValue(value);

    EXPECT_EQ(src.intrinsic_.width_, dst.intrinsic_.width_);
    EXPECT_EQ(src.intrinsic_.height_, dst.intrinsic_.height_);

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            EXPECT_NEAR(src.intrinsic_.intrinsic_matrix_(i, j),
                        dst.intrinsic_.intrinsic_matrix_(i, j), THRESHOLD_1E_6);

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            EXPECT_NEAR(src.extrinsic_[0](i, j),
                        dst.extrinsic_[0](i, j), THRESHOLD_1E_6);
}
