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

#include <gtest/gtest.h>
#include <iostream>

#include <Core/Geometry/Image.h>

using namespace std;

TEST(Image, DefaultConstructor)
{
    // FAIL() << "Not implemented.";
    // ADD_FAILURE() << "Not implemented.";
    // EXPECT_TRUE(false) << "Not implemented";

    cout << "\033[0;32m" << "[          ] " << "\033[0;0m";
    cout << "\033[0;31m" << "Not implemented." << "\033[0;0m" << endl;
    FAIL();

    three::Image image;

    // inherited from Geometry2D
    EXPECT_EQ(three::Geometry::GeometryType::Image, image.GetGeometryType());
    EXPECT_EQ(2, image.Dimension());

    // public member variables
    EXPECT_EQ(0, image.width_);
    EXPECT_EQ(0, image.height_);
    EXPECT_EQ(0, image.num_of_channels_);
    EXPECT_EQ(0, image.bytes_per_channel_);
    EXPECT_EQ(0, image.data_.size());

    // public members
    EXPECT_TRUE(image.IsEmpty());
    EXPECT_EQ(Eigen::Vector2d(0.0, 0.0), image.GetMinBound());
    EXPECT_EQ(Eigen::Vector2d(0.0, 0.0), image.GetMaxBound());

    image.Clear();

    // public member variables
    EXPECT_EQ(0, image.width_);
    EXPECT_EQ(0, image.height_);
    EXPECT_EQ(0, image.num_of_channels_);
    EXPECT_EQ(0, image.bytes_per_channel_);
    EXPECT_EQ(0, image.data_.size());

    // public members
    EXPECT_TRUE(image.IsEmpty());
    EXPECT_EQ(Eigen::Vector2d(0.0, 0.0), image.GetMinBound());
    EXPECT_EQ(Eigen::Vector2d(0.0, 0.0), image.GetMaxBound());

    image.width_ = 1920;
    image.height_ = 1080;
    image.num_of_channels_ = 3;
    image.bytes_per_channel_ = 8;

    // public member variables
    EXPECT_EQ(1920, image.width_);
    EXPECT_EQ(1080, image.height_);
    EXPECT_EQ(3, image.num_of_channels_);
    EXPECT_EQ(8, image.bytes_per_channel_);
    EXPECT_EQ(0, image.data_.size());

    // public members
    EXPECT_TRUE(image.IsEmpty());
    EXPECT_EQ(Eigen::Vector2d(0.0, 0.0), image.GetMinBound());
    EXPECT_EQ(Eigen::Vector2d(1920, 1080), image.GetMaxBound());
}
