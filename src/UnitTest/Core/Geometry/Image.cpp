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
// Initialize a uint8_t vector with random values in the [0:255] range.
// ----------------------------------------------------------------------------
void randInit(vector<uint8_t>& v)
{
    uint8_t vmin = 0;
    uint8_t vmax = 255;
    float factor = (float)(vmax - vmin) / RAND_MAX;

    for (size_t i = 0; i < v.size(); i++)
        v[i] = vmin + (uint8_t)(std::rand() * factor);
}

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
    open3d::Image image;

    image.PrepareImage(default_width,
                       default_height,
                       default_num_of_channels,
                       default_bytes_per_channel);

    // public member variables
    EXPECT_EQ(default_width, image.width_);
    EXPECT_EQ(default_height, image.height_);
    EXPECT_EQ(default_num_of_channels, image.num_of_channels_);
    EXPECT_EQ(default_bytes_per_channel, image.bytes_per_channel_);
    EXPECT_EQ(default_width *
              default_height *
              default_num_of_channels *
              default_bytes_per_channel, image.data_.size());

    // public members
    EXPECT_FALSE(image.IsEmpty());
    EXPECT_TRUE(image.HasData());
    EXPECT_EQ(Eigen::Vector2d(0, 0), image.GetMinBound());
    EXPECT_EQ(Eigen::Vector2d(default_width,
                              default_height), image.GetMaxBound());
    EXPECT_TRUE(image.TestImageBoundary(0, 0));
    EXPECT_EQ(default_width *
              default_num_of_channels *
              default_bytes_per_channel, image.BytesPerLine());
}

// ----------------------------------------------------------------------------
// test Clear
// ----------------------------------------------------------------------------
TEST(Image, Clear)
{
    open3d::Image image;

    image.PrepareImage(default_width,
                       default_height,
                       default_num_of_channels,
                       default_bytes_per_channel);

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
    open3d::Image image;

    const int local_width = 10;
    const int local_height = 10;
    const int local_num_of_channels = 1;
    const int local_bytes_per_channel = 4;

    image.PrepareImage(local_width,
                       local_height,
                       local_num_of_channels,
                       local_bytes_per_channel);

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
    open3d::Image image;

    image.PrepareImage(default_width,
                       default_height,
                       default_num_of_channels,
                       default_bytes_per_channel);

    int temp_width = 320;
    int temp_height = 240;
    int temp_num_of_channels = 1;
    int temp_bytes_per_channel = 3;

    image.width_ = temp_width;
    EXPECT_EQ(temp_width *
              default_height *
              default_num_of_channels *
              default_bytes_per_channel, image.data_.size());

    image.width_ = default_width;
    image.height_ = temp_height;
    EXPECT_EQ(default_width *
              temp_height *
              default_num_of_channels *
              default_bytes_per_channel, image.data_.size());

    image.height_ = default_height;
    image.num_of_channels_ = temp_num_of_channels;
    EXPECT_EQ(default_width *
              default_height *
              temp_num_of_channels *
              default_bytes_per_channel, image.data_.size());

    image.num_of_channels_ = default_num_of_channels;
    image.bytes_per_channel_ = temp_bytes_per_channel;
    EXPECT_EQ(default_width *
              default_height *
              default_num_of_channels *
              temp_bytes_per_channel, image.data_.size());

    image.bytes_per_channel_ = default_bytes_per_channel;
    image.data_ = vector<uint8_t>();
    EXPECT_EQ(default_width *
              default_height *
              default_num_of_channels *
              default_bytes_per_channel, image.data_.size());
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, DISABLED_CreateDepthToCameraDistanceMultiplierFloatImage)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, CreateFloatImageFromImage)
{
    open3d::Image image;

    int local_width = 10;
    int local_height = 10;
    int local_num_of_channels = 1;
    int local_bytes_per_channel = 1;

    image.PrepareImage(local_width,
                       local_height,
                       local_num_of_channels,
                       local_bytes_per_channel);

    randInit(image.data_);
    // for (size_t i = 0; i < image.data_.size(); i++)
    //     {
    //         if (i % local_width == 0)
    //             cout << endl;
    //         cout << setw(5) << (float)image.data_[i];
    //     }
    // cout << endl;
    // cout << endl;

    vector<uint8_t> ref = {
        215, 214,  86,  63, 201, 200, 200,  62, 200, 199,\
         71,  63, 204, 203,  75,  63, 233, 232, 104,  63,\
        201, 200,  72,  62, 171, 170, 170,  62, 196, 195,\
         67,  63, 141, 140, 140,  62, 142, 141,  13,  63,\
        243, 242, 242,  62, 161, 160,  32,  63, 187, 186,\
        186,  62, 131, 130,   2,  63, 243, 242, 114,  63,\
        234, 233, 105,  63, 163, 162,  34,  63, 183, 182,\
         54,  63, 145, 144,  16,  62, 155, 154,  26,  63,\
        129, 128, 128,  60, 245, 244, 116,  62, 137, 136,\
          8,  62, 206, 205,  77,  63, 157, 156,  28,  62,\
        205, 204, 204,  62, 133, 132,   4,  62, 217, 216,\
        216,  61, 255, 254, 126,  63, 221, 220,  92,  62,\
        131, 130,   2,  63, 214, 213,  85,  63, 157, 156,\
         28,  63, 151, 150, 150,  62, 163, 162,  34,  63,\
        134, 133,   5,  63, 251, 250, 250,  62, 249, 248,\
        120,  63, 149, 148, 148,  62, 197, 196,  68,  63,\
        135, 134,   6,  63, 197, 196,  68,  63, 205, 204,\
        204,  62, 228, 227,  99,  63, 145, 144, 144,  62,\
        179, 178, 178,  62, 206, 205,  77,  63, 235, 234,\
        106,  63, 137, 136, 136,  61, 243, 242, 114,  63,\
        135, 134,   6,  63, 169, 168, 168,  61, 197, 196,\
         68,  62, 170, 169,  41,  63, 228, 227,  99,  63,\
        177, 176, 176,  62, 129, 128, 128,  61, 161, 160,\
        160,  60, 233, 232, 232,  62, 129, 128, 128,  61,\
        241, 240, 112,  62, 248, 247, 119,  63, 231, 230,\
        102,  63, 217, 216,  88,  63, 135, 134, 134,  62,\
        138, 137,   9,  63, 191, 190, 190,  62, 194, 193,\
         65,  63, 131, 130,   2,  63, 171, 170,  42,  63,\
        136, 135,   7,  63, 161, 160,  32,  61, 223, 222,\
        222,  62, 238, 237, 109,  63, 238, 237, 109,  63,\
        184, 183,  55,  63, 145, 144, 144,  62, 189, 188,\
         60,  63, 164, 163,  35,  63, 181, 180, 180,  62,\
        176, 175,  47,  63, 169, 168,  40,  62, 225, 224,\
        224,  62, 225, 224,  96,  63, 212, 211,  83,  63,\
        169, 168, 168,  62, 233, 232, 104,  62, 228, 227,\
         99,  63, 179, 178, 178,  62, 176, 175,  47,  63,\
        244, 243, 115,  63, 151, 150,  22,  63, 168, 167,\
         39,  63, 219, 218,  90,  63, 225, 224, 224,  62,\
        236, 235, 107,  63, 203, 202, 202,  62, 208, 207,\
         79,  63, 175, 174,  46,  63, 233, 232, 104,  63, };

    auto floatImage = open3d::CreateFloatImageFromImage(image);
    for (size_t i = 0; i < floatImage->data_.size(); i++)
        {
            if ((i % 10 == 0) && (i != 0))
                cout << "\\" << endl;
            cout << setw(4) << (float)floatImage->data_[i] << ",";
        }
    cout << endl;

    EXPECT_FALSE(floatImage->IsEmpty());
    EXPECT_EQ(local_width, floatImage->width_);
    EXPECT_EQ(local_height, floatImage->height_);
    EXPECT_EQ(local_num_of_channels, floatImage->num_of_channels_);
    EXPECT_EQ(4, floatImage->bytes_per_channel_);
    for (size_t i = 0; i < floatImage->data_.size(); i++)
        EXPECT_EQ(ref[i], floatImage->data_[i]);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, PointerAt)
{
    open3d::Image image;

    const int local_width = 10;
    const int local_height = 10;
    const int local_num_of_channels = 1;
    const int local_bytes_per_channel = 4;

    image.PrepareImage(local_width,
                       local_height,
                       local_num_of_channels,
                       local_bytes_per_channel);

    float* im = reinterpret_cast<float*>(&image.data_[0]);

    im[0 * local_width + 0] = 0.0;
    im[0 * local_width + 1] = 1.0;
    im[1 * local_width + 0] = 2.0;
    im[1 * local_width + 1] = 3.0;

    EXPECT_EQ(0.0, *open3d::PointerAt<float>(image, 0, 0));
    EXPECT_EQ(1.0, *open3d::PointerAt<float>(image, 1, 0));
    EXPECT_EQ(2.0, *open3d::PointerAt<float>(image, 0, 1));
    EXPECT_EQ(3.0, *open3d::PointerAt<float>(image, 1, 1));
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, DISABLED_ConvertDepthToFloatImage)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, DISABLED_FlipImage)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, DISABLED_FilterImage)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, DISABLED_FilterHorizontalImage)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, DISABLED_DownsampleImage)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, DISABLED_DilateImage)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, DISABLED_LinearTransformImage)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, DISABLED_ClipIntensityImage)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, DISABLED_CreateImageFromFloatImage)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, DISABLED_FilterImagePyramid)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, DISABLED_CreateImagePyramid)
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