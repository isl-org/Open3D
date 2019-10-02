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

#include "TestUtility/UnitTest.h"

// #include "Open3D/ColorMap/ColorMapOptimizationOption.h"

/* TODO
As the color_map::ColorMapOptimization subcomponents go back into hiding several
lines of code had to commented out. Do not remove these lines, they may become
useful again after a decision has been made about the way to make these
subcomponents visible to UnitTest.
*/

TEST(ColorMapOptimizationOption, DISABLED_Constructor) {
    // open3d::ColorMapOptimizationOption option;

    // EXPECT_FALSE(option.non_rigid_camera_coordinate_);

    // EXPECT_EQ(16, option.number_of_vertical_anchors_);
    // EXPECT_EQ(3, option.half_dilation_kernel_size_for_discontinuity_map_);

    // EXPECT_NEAR(0.316, option.non_rigid_anchor_point_weight_,
    // unit_test::THRESHOLD_1E_6); EXPECT_NEAR(300, option.maximum_iteration_,
    // unit_test::THRESHOLD_1E_6); EXPECT_NEAR(2.5,
    // option.maximum_allowable_depth_, unit_test::THRESHOLD_1E_6);
    // EXPECT_NEAR(0.03, option.depth_threshold_for_visiblity_check_,
    // unit_test::THRESHOLD_1E_6); EXPECT_NEAR(0.1,
    // option.depth_threshold_for_discontinuity_check_,
    // unit_test::THRESHOLD_1E_6);
}
