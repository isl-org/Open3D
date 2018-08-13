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

static const int default_width = 1920;
static const int default_height = 1080;
static const int default_num_of_channels = 3;
static const int default_bytes_per_channel = 1;

// ----------------------------------------------------------------------------
// test the default constructor scenario
// ----------------------------------------------------------------------------
TEST(Image, DefaultConstructor)
{
    open3d::Image image;

    // inherited from Geometry2D
    EXPECT_EQ(open3d::Geometry::GeometryType::Image, image.GetGeometryType());
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

// ----------------------------------------------------------------------------
// test PrepareImage aka image creation
// ----------------------------------------------------------------------------
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

// ----------------------------------------------------------------------------
// test Clear
// ----------------------------------------------------------------------------
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

// ----------------------------------------------------------------------------
// test FloatValueAt, bilinear(?) interpolation
// ----------------------------------------------------------------------------
TEST(Image, FloatValueAt)
{
    three::Image image;

    const int local_width = 10;
    const int local_height = 10;
    const int local_num_of_channels = 1;
    const int local_bytes_per_channel = 4;

    image.PrepareImage(local_width, local_height, local_num_of_channels, local_bytes_per_channel);

    float* im = reinterpret_cast<float*>(&image.data_[0]);

    im[0 * local_width + 0] = 4;
    im[0 * local_width + 1] = 4;
    im[1 * local_width + 0] = 4;
    im[1 * local_width + 1] = 4;

    EXPECT_EQ(4.0, image.FloatValueAt(0.0, 0.0).second);
    EXPECT_EQ(4.0, image.FloatValueAt(0.0, 1.0).second);
    EXPECT_EQ(4.0, image.FloatValueAt(1.0, 0.0).second);
    EXPECT_EQ(4.0, image.FloatValueAt(1.0, 1.0).second);

    EXPECT_EQ(4.0, image.FloatValueAt(0.5, 0.5).second);

    EXPECT_EQ(2.0, image.FloatValueAt(0.0, 1.5).second);
    EXPECT_EQ(2.0, image.FloatValueAt(1.5, 0.0).second);
    EXPECT_EQ(1.0, image.FloatValueAt(1.5, 1.5).second);
}

// ----------------------------------------------------------------------------
// member data is not private and as such can lead to errors
// ----------------------------------------------------------------------------
TEST(Image, MemberData)
{
    three::Image image;

    image.PrepareImage(default_width, default_height, default_num_of_channels, default_bytes_per_channel);

    int temp_width = 320;
    int temp_height = 240;
    int temp_num_of_channels = 1;
    int temp_bytes_per_channel = 3;

    image.width_ = temp_width;
    EXPECT_EQ(temp_width * default_height * default_num_of_channels * default_bytes_per_channel, image.data_.size());

    image.width_ = default_width;
    image.height_ = temp_height;
    EXPECT_EQ(default_width * temp_height * default_num_of_channels * default_bytes_per_channel, image.data_.size());

    image.height_ = default_height;
    image.num_of_channels_ = temp_num_of_channels;
    EXPECT_EQ(default_width * default_height * temp_num_of_channels * default_bytes_per_channel, image.data_.size());

    image.num_of_channels_ = default_num_of_channels;
    image.bytes_per_channel_ = temp_bytes_per_channel;
    EXPECT_EQ(default_width * default_height * default_num_of_channels * temp_bytes_per_channel, image.data_.size());

    image.bytes_per_channel_ = default_bytes_per_channel;
    image.data_ = vector<uint8_t>();
    EXPECT_EQ(default_width * default_height * default_num_of_channels * default_bytes_per_channel, image.data_.size());
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, CreateDepthToCameraDistanceMultiplierFloatImage)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, CreateFloatImageFromImage)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, PointerAt)
{
    three::Image image;

    const int local_width = 10;
    const int local_height = 10;
    const int local_num_of_channels = 1;
    const int local_bytes_per_channel = 4; 

    image.PrepareImage(local_width, local_height, local_num_of_channels, local_bytes_per_channel);

    float* im = reinterpret_cast<float*>(&image.data_[0]);

    im[0 * local_width + 0] = 0.0;
    im[0 * local_width + 1] = 1.0;
    im[1 * local_width + 0] = 2.0;
    im[1 * local_width + 1] = 3.0;

    EXPECT_EQ(0.0, *three::PointerAt<float>(image, 0, 0));
    EXPECT_EQ(1.0, *three::PointerAt<float>(image, 1, 0));
    EXPECT_EQ(2.0, *three::PointerAt<float>(image, 0, 1));
    EXPECT_EQ(3.0, *three::PointerAt<float>(image, 1, 1));
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, ConvertDepthToFloatImage)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, FlipImage)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, FilterImage)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, FilterHorizontalImage)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, DownsampleImage)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, DilateImage)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, LinearTransformImage)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, ClipIntensityImage)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, CreateImageFromFloatImage)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, FilterImagePyramid)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, CreateImagePyramid)
{
    NotImplemented();
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
- PointerAt doesn't look like it has exception handling.
Ideally all methods should provide some kind of exception handling; at the very least throw exceptions on invalid cases.
- PointerAt should be part of Image::
It really only accepts Image as the first argument.
- Image::FloatValueAt doesn't look like it has exception handling.
Instead it uses 'if' to check the inputs. While this works fine it is generally slower then throwing an exception.
Also, only clipping is used for out of bounds coordinates. There are other accepted methods for dealing with out of bounds (mirroring, wrapping, etc.), do we want to add those as well?
- Image::FloatValueAt returns double not float... even though PointerAt returns float
- Image::FloatValueAt doesn't need to return a pair of bool and the computed value
- Image::BytesPerLine() can be cached
This method is a shortcut for width_ * num_of_channels_ * bytes_per_channel_ which doesn't change often.
- Image:: public data
Ideally make all data private. Add getters/setters as appropriate.
- Image:: needs constructor that creates an image
- Image:: needs copy constructor, equal operator, assignment operator, etc.
(from const reference, from const pointer)
- Image::PrepareImage should be Image::Create
- ImageFactory.cpp not needed.
Can't have global methods.
Find another place for the code, maybe in Image or PinholeCameraIntrinsic.
- The hardcoded filters Gaussian3/5/7 and Sobel31/32 can/should be const arrays instead of vectors.

*/