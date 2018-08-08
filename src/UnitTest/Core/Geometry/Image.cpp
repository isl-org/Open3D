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

#include "Core/Geometry/Image.h"

using namespace std;

static int default_width = 1920;
static int default_height = 1080;
static int default_num_of_channels = 3;
static int default_bytes_per_channel = 1;

// test the default constructor scenario
TEST(Image, DefaultConstructor)
{
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
    EXPECT_FALSE(image.HasData());
    EXPECT_EQ(Eigen::Vector2d(0, 0), image.GetMinBound());
    EXPECT_EQ(Eigen::Vector2d(0, 0), image.GetMaxBound());
    EXPECT_FALSE(image.TestImageBoundary(0, 0));
    EXPECT_EQ(0, image.BytesPerLine());
}

// test PrepareImage
TEST(Image, CreateImage)
{
    three::Image image;

    image.PrepareImage(default_width, default_height, default_num_of_channels, default_bytes_per_channel);

    // public member variables
    EXPECT_EQ(default_width, image.width_);
    EXPECT_EQ(default_height, image.height_);
    EXPECT_EQ(default_num_of_channels, image.num_of_channels_);
    EXPECT_EQ(default_bytes_per_channel, image.bytes_per_channel_);
    EXPECT_EQ(default_width * default_height * default_num_of_channels * default_bytes_per_channel, image.data_.size());

    // public members
    EXPECT_FALSE(image.IsEmpty());
    EXPECT_TRUE(image.HasData());
    EXPECT_EQ(Eigen::Vector2d(0, 0), image.GetMinBound());
    EXPECT_EQ(Eigen::Vector2d(default_width, default_height), image.GetMaxBound());
    EXPECT_TRUE(image.TestImageBoundary(0, 0));
    EXPECT_EQ(default_width * default_num_of_channels * default_bytes_per_channel, image.BytesPerLine());
}

// test Clear
TEST(Image, Clear)
{
    three::Image image;

    image.PrepareImage(default_width, default_height, default_num_of_channels, default_bytes_per_channel);

    image.Clear();

    // public member variables
    EXPECT_EQ(0, image.width_);
    EXPECT_EQ(0, image.height_);
    EXPECT_EQ(0, image.num_of_channels_);
    EXPECT_EQ(0, image.bytes_per_channel_);
    EXPECT_EQ(0, image.data_.size());

    // public members
    EXPECT_TRUE(image.IsEmpty());
    EXPECT_FALSE(image.HasData());
    EXPECT_EQ(Eigen::Vector2d(0, 0), image.GetMinBound());
    EXPECT_EQ(Eigen::Vector2d(0, 0), image.GetMaxBound());
    EXPECT_FALSE(image.TestImageBoundary(0, 0));
    EXPECT_EQ(0, image.BytesPerLine());
}

// member data is not private and as such can lead to errors 
TEST(Image, MemberData)
{
    three::Image image;

    image.PrepareImage(default_width, default_height, default_num_of_channels, default_bytes_per_channel);

    image.width_ = 320;
    EXPECT_EQ(320 * default_height * default_num_of_channels * default_bytes_per_channel, image.data_.size());

    image.width_ = default_width;
    image.height_ = 240;
    EXPECT_EQ(default_width * 240 * default_num_of_channels * default_bytes_per_channel, image.data_.size());

    image.height_ = default_height;
    image.num_of_channels_ = 1;
    EXPECT_EQ(default_width * default_height * 1 * default_bytes_per_channel, image.data_.size());

    image.num_of_channels_ = default_num_of_channels;
    image.bytes_per_channel_ = 3;
    EXPECT_EQ(default_width * default_height * default_num_of_channels * 3, image.data_.size());

    image.bytes_per_channel_ = default_bytes_per_channel;
    image.data_ = vector<uint8_t>();
    EXPECT_EQ(default_width * default_height * default_num_of_channels * default_bytes_per_channel, image.data_.size());
}

// NOTES
/*
- Image:: not enough comments; for example, what is the intent of Image::TestImageBoundary(...)?
- Template definitions should be placed in the header file.
After that specializations can be defined in the source file.
- dependency on the Eigen library for its datatypes
Ideally we would use our own types and provide a method to convert to Eigen types if necessary.
This would open the posibility to replace Eigen at any time with another library, possibly GPU supported.
- still using the namespace three; open3d would be more appropriate given the extensive branding process we've gone through. 
- lots of functions declared/defined in the root of the global namespace, outside the Image class
- public data members
- base class Geometry/2D/3D has nothing to do with geometry
- Geometry/2D/3D are nothing but interfaces, could very well be named something else
- Geometry sets dimension to 3 although itself is dimensionless
- Geometry2D vs Geometry3d
  - dimension 2 vs 3
  - no transform vs supports transform
In reality we should have a transform function supported by Geometry2D as well.
In fact the same Transform(Matrix4d) function should be supported by both Geometry2D and Geometry3D.
For Geometry2D maybe we can simply ignore the Z component of the computation.
- FilterType should not be a property of Image; it's a property of a filter, duh
Ideally all Open3D types should be defined in a central location (aka types file).
- Geometry::GeometryType could very well be moved to a types file.
-  Image.h has inline definitions
Ideally no definitions inside the header, only declarations (and template defs)
- Image::HasData(...) and Image::IsEmpty(...) are the same
Ideally keep just one.
- Image::FloatValueAt doesn't look like it has exception handling
Ideally all methods should provide some kind of exception handling; at the very least throw exceptions on invalid cases.
- Image::BytesPerLine() can be cached
This method is a shortcut for width_ * num_of_channels_ * bytes_per_channel_.
- Image:: public data
Ideally make all data private. Add getters/setters as appropriate.
- Image:: needs constructor that creates an image
- Image:: needs copy constructor, equal operator, assignment operator, etc.
(from const reference, from const pointer)
- Image::PrepareImage should be Image::Create
*/