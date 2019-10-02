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

#include <json/json.h>

#include "Open3D/Camera/PinholeCameraTrajectory.h"
#include "TestUtility/UnitTest.h"

using namespace Eigen;
using namespace open3d;
using namespace std;
using namespace unit_test;

TEST(PinholeCameraTrajectory, DISABLED_MemberData) {
    unit_test::NotImplemented();
}

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
        intrinsic.intrinsic_matrix_ = Matrix3d::Random();

        src.parameters_[i].intrinsic_ = intrinsic;
        src.parameters_[i].extrinsic_ = Matrix4d::Random();
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
