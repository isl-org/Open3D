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

using namespace Eigen;
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
    PinholeCameraTrajectory src;
    PinholeCameraTrajectory dst;

    int width = 640;
    int height = 480;

    src.intrinsic_.width_ = width;
    src.intrinsic_.height_ = height;
    src.intrinsic_.intrinsic_matrix_ = Matrix3d::Random();

    src.extrinsic_.push_back(Matrix4d::Random());

    Json::Value value;
    bool output = src.ConvertToJsonValue(value);
    EXPECT_TRUE(output);

    output = dst.ConvertFromJsonValue(value);
    EXPECT_TRUE(output);

    EXPECT_EQ(src.intrinsic_.width_, dst.intrinsic_.width_);
    EXPECT_EQ(src.intrinsic_.height_, dst.intrinsic_.height_);

    ExpectEQ(src.intrinsic_.intrinsic_matrix_,
             dst.intrinsic_.intrinsic_matrix_);

    ExpectEQ(src.extrinsic_[0], dst.extrinsic_[0]);
}
