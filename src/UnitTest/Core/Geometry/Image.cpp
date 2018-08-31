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
#include "Core/Camera/PinholeCameraIntrinsic.h"

using namespace std;

static const int default_width = 1920;
static const int default_height = 1080;
static const int default_num_of_channels = 3;
static const int default_bytes_per_channel = 1;

// ----------------------------------------------------------------------------
// Initialize a uint8_t vector with random values in the [0:255] range.
// ----------------------------------------------------------------------------
void RandInit(vector<uint8_t>& v)
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

    im[0 * local_width + 0] = 4.0f;
    im[0 * local_width + 1] = 4.0f;
    im[1 * local_width + 0] = 4.0f;
    im[1 * local_width + 1] = 4.0f;

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
TEST(Image, CreateDepthToCameraDistanceMultiplierFloatImage)
{
    open3d::PinholeCameraIntrinsic intrinsic = 
        open3d::PinholeCameraIntrinsic(
            open3d::PinholeCameraIntrinsicParameters::PrimeSenseDefault);

    auto image = CreateDepthToCameraDistanceMultiplierFloatImage(intrinsic);

    // test image dimensions
    const int local_width = 640;
    const int local_height = 480;
    const int local_num_of_channels = 1;
    const int local_bytes_per_channel = 4;

    EXPECT_FALSE(image->IsEmpty());
    EXPECT_EQ(local_width, image->width_);
    EXPECT_EQ(local_height, image->height_);
    EXPECT_EQ(local_num_of_channels, image->num_of_channels_);
    EXPECT_EQ(local_bytes_per_channel, image->bytes_per_channel_);
}

// ----------------------------------------------------------------------------
// Tests one of the following configurations
// channels: bytes per channel
// 1: 1/2/4
// 3: 1/2/4 with either Equal or Weighted type
// ----------------------------------------------------------------------------
void TEST_CreateFloatImageFromImage(
    const int& num_of_channels,
    const int& bytes_per_channel,
    const vector<uint8_t>& ref,
    const open3d::Image::ColorToIntensityConversionType& type)
{
    open3d::Image image;

    // test image dimensions
    const int local_width = 5;
    const int local_height = 5;
    const int float_num_of_channels = 1;

    image.PrepareImage(local_width,
                       local_height,
                       num_of_channels,
                       bytes_per_channel);

    RandInit(image.data_);

    auto floatImage = open3d::CreateFloatImageFromImage(image);

    EXPECT_FALSE(floatImage->IsEmpty());
    EXPECT_EQ(local_width, floatImage->width_);
    EXPECT_EQ(local_height, floatImage->height_);
    EXPECT_EQ(float_num_of_channels, floatImage->num_of_channels_);
    EXPECT_EQ(sizeof(float), floatImage->bytes_per_channel_);
    for (size_t i = 0; i < floatImage->data_.size(); i++)
        EXPECT_EQ(ref[i], floatImage->data_[i]);
}

// ----------------------------------------------------------------------------
// Tests one of the following configuration:
// channels: 1
// bytes per channel: 1
// ----------------------------------------------------------------------------
TEST(Image, CreateFloatImageFromImage_1_1)
{
    // reference data used to validate the creation of the float image
    vector<uint8_t> ref = { 215, 214,  86,  63, 201, 200, 200,  62, 200, 199,\
                             71,  63, 204, 203,  75,  63, 233, 232, 104,  63,\
                            201, 200,  72,  62, 171, 170, 170,  62, 196, 195,\
                             67,  63, 141, 140, 140,  62, 142, 141,  13,  63,\
                            243, 242, 242,  62, 161, 160,  32,  63, 187, 186,\
                            186,  62, 131, 130,   2,  63, 243, 242, 114,  63,\
                            234, 233, 105,  63, 163, 162,  34,  63, 183, 182,\
                             54,  63, 145, 144,  16,  62, 155, 154,  26,  63,\
                            129, 128, 128,  60, 245, 244, 116,  62, 137, 136,\
                              8,  62, 206, 205,  77,  63, 157, 156,  28,  62 };

    TEST_CreateFloatImageFromImage(1, 1, ref, open3d::Image::ColorToIntensityConversionType::Weighted);
}

// ----------------------------------------------------------------------------
// Tests one of the following configuration:
// channels: 1
// bytes per channel: 2
// ----------------------------------------------------------------------------
TEST(Image, CreateFloatImageFromImage_1_2)
{
    // reference data used to validate the creation of the float image
    vector<uint8_t> ref = {    0, 152,   5,  70,   0,  27, 126,  71,   0,  55,\
                               2,  71,   0, 213,  28,  71,   0,  75,  34,  71,\
                               0,  10, 251,  70,   0, 240, 149,  70,   0, 196,\
                               6,  71,   0, 136, 205,  70,   0, 198, 145,  70,\
                               0,  89,  77,  71,   0,  80, 143,  69,   0, 242,\
                               6,  71,   0,  84,  68,  70,   0, 169,  99,  71,\
                               0, 192, 130,  69,   0,  10, 232,  70,   0,  64,\
                             112,  70,   0, 247, 102,  71,   0, 176, 135,  70,\
                               0,  18, 191,  70,   0, 193,   2,  71,   0, 170,\
                               7,  71,   0,  20, 222,  70,   0, 237, 109,  71 };

    TEST_CreateFloatImageFromImage(1, 2, ref, open3d::Image::ColorToIntensityConversionType::Weighted);
}

// ----------------------------------------------------------------------------
// Tests one of the following configuration:
// channels: 1
// bytes per channel: 4
// ----------------------------------------------------------------------------
TEST(Image, CreateFloatImageFromImage_1_4)
{
    // reference data used to validate the creation of the float image
    vector<uint8_t> ref = { 183,  72, 188, 163,  90, 175,  42, 112, 224, 211,\
                             84,  58, 227,  89, 175, 243, 150, 167, 218, 112,\
                            235, 101, 207, 174, 232, 123,  55, 242, 234,  37,\
                            224, 163, 110, 157,  71, 200,  78, 113,  57,  47,\
                             70, 141, 106,  43, 231,  26,  32, 126, 193, 251,\
                            238, 174,  97, 191,  94,  75,  59, 149,  62,  38,\
                            186,  31, 202,  41, 189,  19, 242,  13, 132,  44,\
                             61, 203, 186, 167, 246, 163, 193,  23,  34, 132,\
                             19,  17,  52, 117, 209, 146, 192,  13,  40, 254,\
                             52, 226,  31, 254,  13, 221,  18,   1, 235, 151 };

    TEST_CreateFloatImageFromImage(1, 4, ref, open3d::Image::ColorToIntensityConversionType::Weighted);
}

// ----------------------------------------------------------------------------
// Tests one of the following configuration:
// channels: 3
// bytes per channel: 1
// ColorToIntensityConversionType: Weighted
// ----------------------------------------------------------------------------
TEST(Image, CreateFloatImageFromImage_3_1_Weighted)
{
    // reference data used to validate the creation of the float image
    vector<uint8_t> ref = { 100, 255,  67,  62,   4,  56,  75,  63,  30, 208,\
                             13,  63,  27, 109, 165,  62, 193,  67,   4,  63,\
                            223,  72,  69,  63,  13,  39, 150,  62, 161, 135,\
                             99,  63, 255, 179, 103,  63,  56, 210, 107,  63,\
                             77,  67, 172,  62, 200, 214, 195,  62, 154, 172,\
                            239,  62, 224,  37, 220,  62, 111, 249,  58,  62,\
                            228, 207,   1,  63,  37, 228,   3,  63, 212,  84,\
                            170,  62,  53, 206,  71,  63, 160, 100,  37,  63,\
                            153,  84, 137,  62,  58,  46, 236,  62, 201,  65,\
                            235,  62,  36,  11, 199,  62, 104,  69, 170,  62  };

    TEST_CreateFloatImageFromImage(3, 1, ref, open3d::Image::ColorToIntensityConversionType::Weighted);
}

// ----------------------------------------------------------------------------
// Tests one of the following configuration:
// channels: 3
// bytes per channel: 1
// ColorToIntensityConversionType: Equal
// ----------------------------------------------------------------------------
TEST(Image, CreateFloatImageFromImage_3_1_Equal)
{
    // reference data used to validate the creation of the float image
    vector<uint8_t> ref = {  37, 222,  28,  63,  67,  75, 247,  62,  57, 133,\
                            252,  62, 135,  30,  45,  63, 209,  42, 207,  62,\
                            242,   9, 142,  62,  27, 250, 137,  62,  87, 134,\
                              3,  63, 172, 150, 232,  62, 154, 208, 153,  62,\
                             31, 193, 198,  62,  10,  59, 131,  62, 184, 116,\
                            105,  63,  16,  12, 182,  62,  48, 173,  44,  63,\
                            138, 213,  76,  63, 161,  33,  23,  63,  96, 147,\
                            102,  63,  50, 218,   0,  63,  10, 170,  36,  63,\
                             66,  33,  49,  63,   9, 120, 138,  62, 246,  85,\
                            187,  62, 229,  25,  56,  63, 127,  44, 116,  62 };

    TEST_CreateFloatImageFromImage(3, 1, ref, open3d::Image::ColorToIntensityConversionType::Equal);
}

// ----------------------------------------------------------------------------
// Tests one of the following configuration:
// channels: 3
// bytes per channel: 2
// ColorToIntensityConversionType: Weighted
// ----------------------------------------------------------------------------
TEST(Image, CreateFloatImageFromImage_3_2_Weighted)
{
    // reference data used to validate the creation of the float image
    vector<uint8_t> ref = { 130, 144, 170,  70, 147, 191,  21,  71,  76,  43,\
                            144,  70,  39, 255, 178,  70,  93, 239,  28,  71,\
                            196,  97,  33,  71,  73, 238, 241,  70, 127,  36,\
                             93,  71,  28, 124,   0,  71, 182, 133,  29,  71,\
                            206, 181, 178,  70, 118, 115, 165,  70, 152,  24,\
                            175,  70, 248,  56, 209,  70, 185, 167,  17,  71,\
                             73,  66,  29,  71, 206, 235, 252,  70,  39, 186,\
                            213,  70,  96, 201,  96,  70, 166, 114, 198,  70,\
                             40,  16, 226,  70,  41, 121, 163,  70,  60, 255,\
                             36,  71, 139,  94,  18,  71, 175, 119, 234,  70 };

    TEST_CreateFloatImageFromImage(3, 2, ref, open3d::Image::ColorToIntensityConversionType::Weighted);
}

// ----------------------------------------------------------------------------
// Tests one of the following configuration:
// channels: 3
// bytes per channel: 2
// ColorToIntensityConversionType: Equal
// ----------------------------------------------------------------------------
TEST(Image, CreateFloatImageFromImage_3_2_Equal)
{
    // reference data used to validate the creation of the float image
    vector<uint8_t> ref = {  14,  69, 211,  69,  87,  14, 122,  70, 132, 220,\
                            193,  70, 147, 187,  58,  71, 115,  16,  82,  71,\
                            166, 203,  55,  71, 164, 243,  15,  71,   0, 192,\
                             59,  71, 166, 116,  56,  71,  34,  11, 227,  70,\
                             16,  14,  18,  71, 238, 210,  19,  71, 187,  78,\
                            190,  70, 133,  97,  33,  71,  34, 225,  23,  70,\
                             34, 250, 234,  70, 171, 124, 200,  70,  38, 153,\
                             21,  71, 176, 217, 250,  70, 204,  50,  82,  71,\
                            135, 240,  88,  71, 188,  77, 255,  70,  15,   8,\
                             12,  71,  75, 129,  12,  71,  69,  63, 154,  70 };

    TEST_CreateFloatImageFromImage(3, 2, ref, open3d::Image::ColorToIntensityConversionType::Equal);
}

// ----------------------------------------------------------------------------
// Tests one of the following configuration:
// channels: 3
// bytes per channel: 4
// ColorToIntensityConversionType: Weighted
// ----------------------------------------------------------------------------
TEST(Image, CreateFloatImageFromImage_3_4_Weighted)
{
    // reference data used to validate the creation of the float image
    vector<uint8_t> ref = {  50, 221, 220, 122,   7,  25, 180, 239,  18, 255,\
                            189, 182,  28,  59,  66, 233, 205, 215,  47,  89,\
                            135, 136, 248, 225,  74, 190, 207, 229, 153,  20,\
                             88, 149,  29,  64, 149, 206, 229,   4, 109,  51,\
                            117, 110, 124, 217, 137, 106,  43,  86,  94, 169,\
                             83, 209, 170, 212, 155, 119, 184, 101, 140,  68,\
                            145, 248,  68,  86, 166, 152, 142, 165,  77,   2,\
                            239,  99,  17, 108,  73,  77, 124, 183, 203, 235,\
                            125,   1, 139, 208, 203, 196,  14, 216,  65, 173,\
                            104, 215, 133,  61,  57, 121, 158, 138,  61, 196 };

    TEST_CreateFloatImageFromImage(3, 4, ref, open3d::Image::ColorToIntensityConversionType::Weighted);
}

// ----------------------------------------------------------------------------
// Tests one of the following configuration:
// channels: 3
// bytes per channel: 4
// ColorToIntensityConversionType: Equal
// ----------------------------------------------------------------------------
TEST(Image, CreateFloatImageFromImage_3_4_Equal)
{
    // reference data used to validate the creation of the float image
    vector<uint8_t> ref = { 145, 227, 199, 110,  27, 234, 201,  98,  13, 245,\
                             74, 117,  64,  93, 208, 218,  19, 100, 163, 243,\
                            194, 245, 203, 117, 100, 223, 161, 115, 159, 244,\
                             59, 223,   6,   2, 149, 106, 243, 174, 111,  79,\
                             45, 249, 167, 196, 177, 141, 146, 103, 180, 154,\
                             85, 235,  92,  62, 123, 182, 162, 190, 102,  91,\
                            180, 148,  90,  97, 158,  31, 130,  70,  17,  25,\
                            136,  98,  47, 202,   3, 127,  79, 211,  16, 200,\
                            125, 201,   4, 198,  33,  89, 126,  60, 107, 179,\
                            232, 113, 198,   8,  44, 219, 146, 174,  55, 241 };

    TEST_CreateFloatImageFromImage(3, 4, ref, open3d::Image::ColorToIntensityConversionType::Equal);
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
TEST(Image, ConvertDepthToFloatImage)
{
    // reference data used to validate the creation of the float image
    vector<uint8_t> ref = {  22, 236,  15,  57, 203,   3,  56,  58, 153, 156,\
                            114,  57,  38,  66,  28,  58,   5, 150,   3,  55,\
                            200, 211,  85,  58,  73, 185, 246,  57, 225, 185,\
                              8,  58, 147, 161,  78,  58,   5, 150,   3,  57,\
                            230, 180,  44,  57, 227, 132,  74,  58, 153, 156,\
                            114,  58, 110, 250,  17,  58, 118,  37, 152,  57,\
                            173, 135, 129,  57, 183, 125,  73,  57,   5, 150,\
                              3,  56, 136,  70, 230,  56, 211,  46,  62,  58,\
                              2, 102,  33,  58,  40,  13,  94,  57, 151, 209,\
                             48,  58,  29, 178, 117,  58, 147, 161,  78,  58 };

    open3d::Image image;

    // test image dimensions
    const int local_width = 5;
    const int local_height = 5;
    const int local_num_of_channels = 1;
    const int local_bytes_per_channel = 1;
    const int float_num_of_channels = 1;

    image.PrepareImage(local_width,
                       local_height,
                       local_num_of_channels,
                       local_bytes_per_channel);

    RandInit(image.data_);

    auto floatImage = open3d::ConvertDepthToFloatImage(image);

    EXPECT_FALSE(floatImage->IsEmpty());
    EXPECT_EQ(local_width, floatImage->width_);
    EXPECT_EQ(local_height, floatImage->height_);
    EXPECT_EQ(float_num_of_channels, floatImage->num_of_channels_);
    EXPECT_EQ(sizeof(float), floatImage->bytes_per_channel_);
    for (size_t i = 0; i < floatImage->data_.size(); i++)
        EXPECT_EQ(ref[i], floatImage->data_[i]);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, FlipImage)
{
    // reference data used to validate the creation of the float image
    vector<uint8_t> ref = { 214, 175,  83, 159, 124, 227, 204, 214, 123, 177,\
                            226,  31,  95,  22, 243, 245,  39,  68, 237,  24,\
                            154,  25,  83,   3,   0,   0,   0,   0,   0,   0,\
                              0,   0,   0,   0,   0,   0,  87, 242,  88, 246,\
                            119,   9,  36,  20, 105, 163,  24,  31, 160, 164,\
                             70, 191,   0,   0,   0,   0,   0,   0,   0,   0,\
                              0,   0,   0,   0,   0,   0,   0,   0, 217, 134,\
                            101, 234,  80, 212,  13, 230,   0,   0,   0,   0,\
                            102,  12,  31,  60,   0,   0,   0,   0,   0,   0,\
                              0,   0,   0,   0,   0,   0, 155, 180, 219,  12 };

    open3d::Image image;

    // test image dimensions
    const int local_width = 5;
    const int local_height = 5;
    const int local_num_of_channels = 1;
    const int local_bytes_per_channel = 4;
    const int flip_bytes_per_channel = 1;

    image.PrepareImage(local_width,
                       local_height,
                       local_num_of_channels,
                       local_bytes_per_channel);

    RandInit(image.data_);

    auto flipImage = open3d::ConvertDepthToFloatImage(image);

    EXPECT_FALSE(flipImage->IsEmpty());
    EXPECT_EQ(local_width, flipImage->width_);
    EXPECT_EQ(local_height, flipImage->height_);
    EXPECT_EQ(flip_bytes_per_channel, flipImage->num_of_channels_);
    EXPECT_EQ(sizeof(float), flipImage->bytes_per_channel_);
    for (size_t i = 0; i < flipImage->data_.size(); i++)
        EXPECT_EQ(ref[i], flipImage->data_[i]);
}

// ----------------------------------------------------------------------------
// Tests one of the following configurations
// channels: bytes per channel
// 1: 1/2/4
// 3: 1/2/4 with either Equal or Weighted type
// ----------------------------------------------------------------------------
void TEST_FilterImage(const vector<uint8_t>& ref,
                 const open3d::Image::FilterType& filter)
{
    open3d::Image image;

    // test image dimensions
    const int local_width = 5;
    const int local_height = 5;
    const int local_num_of_channels = 1;
    const int local_bytes_per_channel = 4;

    image.PrepareImage(local_width,
                       local_height,
                       local_num_of_channels,
                       local_bytes_per_channel);

    RandInit(image.data_);

    auto floatImage = open3d::CreateFloatImageFromImage(image);

    auto outputImage = open3d::FilterImage(*floatImage, filter);

    EXPECT_FALSE(outputImage->IsEmpty());
    EXPECT_EQ(local_width, outputImage->width_);
    EXPECT_EQ(local_height, outputImage->height_);
    EXPECT_EQ(local_num_of_channels, outputImage->num_of_channels_);
    EXPECT_EQ(local_bytes_per_channel, outputImage->bytes_per_channel_);
    for (size_t i = 0; i < outputImage->data_.size(); i++)
        EXPECT_EQ(ref[i], outputImage->data_[i]);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, FilterImage_Gaussian3)
{
    // reference data used to validate the filtering of an image
    vector<uint8_t> ref = {  23, 132,  43, 121, 229, 116, 101, 120,  21,  34,\
                            236, 115, 176, 196, 134,  98, 178, 246,  53,  62,\
                             23, 132, 171, 121, 229, 116, 229, 120,  21,  34,\
                            108, 116, 181, 176, 133, 103,  83, 109,   5, 103,\
                             23, 132,  43, 121, 217, 116, 101, 120, 220, 249,\
                            194, 118, 229,  73,  63, 119,  22,  74, 191, 118,\
                            242, 103,   1, 235, 195, 197, 194, 238,  84,  73,\
                             63, 119, 229,  73, 191, 119,  22,  74,  63, 119,\
                            235,  27, 194, 235,  30, 117,  68, 238,  84,  73,\
                            191, 118, 229,  73,  63, 119,  22,  74, 191, 118 };

    TEST_FilterImage(ref, open3d::Image::FilterType::Gaussian3);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, FilterImage_Gaussian5)
{
    // reference data used to validate the filtering of an image
    vector<uint8_t> ref = { 185, 235,  13, 120,  30, 210, 174, 120, 132, 184,\
                            207, 120, 233, 158, 112, 120,  12, 209,  90, 119,\
                            122, 102, 243, 119, 107, 216, 193, 120, 205, 254,\
                              4, 121, 100,  17, 169, 120, 201,  28, 164, 119,\
                            171,  99, 126, 120,  54,  68, 123, 121,  26,  93,\
                            187, 121,  14,   6, 121, 121,  50, 135, 120, 120,\
                             80,  29, 115, 121, 191, 103, 117, 122, 174,  62,\
                            184, 122, 239, 178, 117, 122, 239, 178, 117, 121,\
                             32, 213,   3, 122, 170,  23,   5, 123,  64, 217,\
                             71, 123,   4,  65,   5, 123,   4,  65,   5, 122 };

    TEST_FilterImage(ref, open3d::Image::FilterType::Gaussian5);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, FilterImage_Gaussian7)
{
    // reference data used to validate the filtering of an image
    vector<uint8_t> ref = { 167, 206,  54, 243, 177,  62, 130, 245, 193,  88,\
                            137, 246, 103,  60,  44, 247, 114, 153, 152, 247,\
                            228, 189, 182, 243,  36,  92,  91, 246, 130, 154,\
                            109, 247, 213,  43,  22, 248,   4, 100, 133, 248,\
                            235, 228, 234, 243, 215,  71, 215, 246, 239,  74,\
                            236, 247, 152, 234, 149, 248, 181,  83,   5, 249,\
                            217, 171, 182, 243, 206, 135,   8, 247, 169,  77,\
                             23, 248,  68, 132, 192, 248, 184,  92,  43, 249,\
                              9, 168,  54, 243, 142, 203, 210, 246, 231, 217,\
                            234, 247, 216, 162, 149, 248, 198,  65,   5, 249 };

    TEST_FilterImage(ref, open3d::Image::FilterType::Gaussian7);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, FilterImage_Sobel3Dx)
{
    // reference data used to validate the filtering of an image
    vector<uint8_t> ref = { 151, 220, 164, 217,  80, 237,  59, 228, 151, 220,\
                            164,  89, 249, 116,   3, 114, 249, 116,   3, 114,\
                            182, 215,  24,  99,  88, 210, 168, 228,  59,  51,\
                            109, 230, 249, 116, 131, 114, 249, 116, 131, 114,\
                            116, 116,  62, 239, 119, 116,  62, 239,  59,  51,\
                            237, 230, 162, 236, 214, 253, 162, 236, 214, 253,\
                            117, 116, 190, 239, 117, 116, 190, 239,  59,  51,\
                            109, 230, 163, 236,  86, 254, 163, 236,  86, 254,\
                            117, 116,  62, 239, 117, 116,  62, 239, 150, 144,\
                            236,  86, 163, 236, 214, 253, 163, 236, 214, 253 };

    TEST_FilterImage(ref, open3d::Image::FilterType::Sobel3Dx);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, FilterImage_Sobel3Dy)
{
    // reference data used to validate the filtering of an image
    vector<uint8_t> ref = {  44, 111,  91, 216,  29,  74, 146, 215, 128,  32,\
                            211, 184, 172, 215, 210, 100, 193,  33, 158, 101,\
                            206,  60, 133,  77, 116, 210, 248,  77, 160,  74,\
                            117,  77,  91, 215, 210, 100, 133,  33, 158, 101,\
                            151, 204, 100, 122, 101, 136, 152, 121, 208,  33,\
                            164, 234,  80, 102, 231,  87,  26, 143, 173,  88,\
                             62,  80, 117, 205, 143,  75, 245, 205, 249,   4,\
                             52, 213, 213,  66,  33,  92,  72, 235, 241,  92,\
                            151, 204, 100, 250, 101, 136, 152, 249, 208,  33,\
                            164, 106, 116, 199, 179, 213, 116, 199,  51, 213 };

    TEST_FilterImage(ref, open3d::Image::FilterType::Sobel3Dy);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, FilterHorizontalImage)
{
    // reference data used to validate the filtering of an image
    vector<uint8_t> ref = { 197, 212, 103,  82, 189, 134,   3, 238, 135, 214,\
                            214, 239,  68,  45,  62, 240, 216, 244, 181, 239,\
                              6, 122,  96,  18,  28,  35, 230,  18, 173,  39,\
                             27, 115, 173,  39, 155, 115, 173,  39,  27, 115,\
                             72, 216, 128, 226, 114,  83,  24, 253, 114,  83,\
                            152, 253, 114,  83,  24, 253,  60,  40, 144,  42,\
                             50, 134, 214, 206,  33,   4,  15, 206, 174,  13,\
                            100, 185, 242, 113,  93, 101, 118,  21,  38, 102,\
                            201, 245,  45,  11, 107, 152, 150,  16, 113, 166,\
                            224, 157, 113, 166,  96, 158, 113, 166, 224, 157 };

    open3d::Image image;

    // test image dimensions
    const int local_width = 5;
    const int local_height = 5;
    const int local_num_of_channels = 1;
    const int local_bytes_per_channel = 4;

    image.PrepareImage(local_width,
                       local_height,
                       local_num_of_channels,
                       local_bytes_per_channel);

    RandInit(image.data_);

    auto floatImage = open3d::CreateFloatImageFromImage(image);

    const std::vector<double> Gaussian3 = { 0.25, 0.5, 0.25 };

    auto outputImage = open3d::FilterHorizontalImage(*floatImage, Gaussian3);

    EXPECT_FALSE(outputImage->IsEmpty());
    EXPECT_EQ(local_width, outputImage->width_);
    EXPECT_EQ(local_height, outputImage->height_);
    EXPECT_EQ(local_num_of_channels, outputImage->num_of_channels_);
    EXPECT_EQ(local_bytes_per_channel, outputImage->bytes_per_channel_);
    for (size_t i = 0; i < outputImage->data_.size(); i++)
        EXPECT_EQ(ref[i], outputImage->data_[i]);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, DownsampleImage)
{
    // reference data used to validate the filtering of an image
    vector<uint8_t> ref = {  14, 215, 190,  56,  52, 246,  60, 113, 234,  61,\
                             44, 103, 184, 214, 134,  90 };

    open3d::Image image;

    // test image dimensions
    const int local_width = 5;
    const int local_height = 5;
    const int local_num_of_channels = 1;
    const int local_bytes_per_channel = 4;

    image.PrepareImage(local_width,
                       local_height,
                       local_num_of_channels,
                       local_bytes_per_channel);

    RandInit(image.data_);

    auto floatImage = open3d::CreateFloatImageFromImage(image);

    auto outputImage = open3d::DownsampleImage(*floatImage);

    EXPECT_FALSE(outputImage->IsEmpty());
    EXPECT_EQ((int)(local_width / 2), outputImage->width_);
    EXPECT_EQ((int)(local_height / 2), outputImage->height_);
    EXPECT_EQ(local_num_of_channels, outputImage->num_of_channels_);
    EXPECT_EQ(local_bytes_per_channel, outputImage->bytes_per_channel_);
    for (size_t i = 0; i < outputImage->data_.size(); i++)
        EXPECT_EQ(ref[i], outputImage->data_[i]);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, DilateImage)
{
    // reference data used to validate the filtering of an image
    vector<uint8_t> ref = { 255, 255,   0,   0,   0,   0,   0, 255, 255, 255,\
                            255, 255,   0,   0,   0,   0, 255, 255, 255, 255,\
                              0,   0,   0,   0,   0, 255, 255, 255, 255, 255,\
                              0,   0,   0,   0, 255, 255, 255, 255, 255,   0,\
                              0,   0,   0, 255, 255, 255, 255, 255,   0,   0,\
                              0,   0, 255, 255, 255, 255, 255,   0,   0,   0,\
                              0, 255, 255, 255, 255, 255,   0,   0,   0,   0,\
                            255, 255, 255, 255, 255,   0,   0,   0,   0,   0,\
                            255, 255, 255, 255,   0,   0,   0,   0, 255, 255,\
                            255, 255, 255,   0,   0,   0,   0,   0, 255, 255 };

    open3d::Image image;

    // test image dimensions
    const int local_width = 10;
    const int local_height = 10;
    const int local_num_of_channels = 1;
    const int local_bytes_per_channel = 1;

    image.PrepareImage(local_width,
                       local_height,
                       local_num_of_channels,
                       local_bytes_per_channel);

    RandInit(image.data_);
    for (size_t i = 0; i < image.data_.size(); i++)
        if (i % 9 == 0)
            image.data_[i] = 255;

    auto outputImage = open3d::DilateImage(image);

    EXPECT_FALSE(outputImage->IsEmpty());
    EXPECT_EQ(local_width, outputImage->width_);
    EXPECT_EQ(local_height, outputImage->height_);
    EXPECT_EQ(local_num_of_channels, outputImage->num_of_channels_);
    EXPECT_EQ(local_bytes_per_channel, outputImage->bytes_per_channel_);
    for (size_t i = 0; i < outputImage->data_.size(); i++)
        EXPECT_EQ(ref[i], outputImage->data_[i]);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, LinearTransformImage)
{
    // reference data used to validate the filtering of an image
    vector<uint8_t> ref = { 120,  93, 188, 208,   3,  75,  74, 127, 154, 153,\
                             25,  62, 154, 153,  25,  62, 154, 153,  25,  62,\
                            133, 147,  25,  62, 155,  11, 228, 103, 239, 208,\
                             23,  72, 154, 153,  25,  62,  32, 128,  18, 249,\
                             66,  66, 195, 251, 154, 153,  25,  62, 176, 245,\
                            254,  61,  24, 100,  81,  60,  96, 145,  49,  62,\
                            154, 153,  25,  62,  95,  77,  38, 228, 182, 137,\
                             25,  62,  46,  10,  79, 218, 154, 153,  25,  62,\
                             62, 190,  94, 109, 208,  15,  24,  62, 149, 153,\
                             25,  62, 187, 207, 224,  80, 154, 153,  25,  62 };

    open3d::Image image;

    // test image dimensions
    const int local_width = 5;
    const int local_height = 5;
    const int local_num_of_channels = 1;
    const int local_bytes_per_channel = 4;

    image.PrepareImage(local_width,
                       local_height,
                       local_num_of_channels,
                       local_bytes_per_channel);

    RandInit(image.data_);

    auto outputImage = open3d::CreateFloatImageFromImage(image);

    open3d::LinearTransformImage(*outputImage, 2.3, 0.15);

    EXPECT_FALSE(outputImage->IsEmpty());
    EXPECT_EQ(local_width, outputImage->width_);
    EXPECT_EQ(local_height, outputImage->height_);
    EXPECT_EQ(local_num_of_channels, outputImage->num_of_channels_);
    EXPECT_EQ(local_bytes_per_channel, outputImage->bytes_per_channel_);
    for (size_t i = 0; i < outputImage->data_.size(); i++)
        EXPECT_EQ(ref[i], outputImage->data_[i]);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, ClipIntensityImage)
{
    // reference data used to validate the filtering of an image
    vector<uint8_t> ref = { 195, 245, 168,  62, 195, 245, 168,  62, 143, 194,\
                             53,  63, 195, 245, 168,  62, 195, 245, 168,  62,\
                            143, 194,  53,  63, 195, 245, 168,  62, 195, 245,\
                            168,  62, 195, 245, 168,  62, 195, 245, 168,  62,\
                            143, 194,  53,  63, 195, 245, 168,  62, 195, 245,\
                            168,  62, 143, 194,  53,  63, 195, 245, 168,  62,\
                            143, 194,  53,  63, 195, 245, 168,  62, 195, 245,\
                            168,  62, 143, 194,  53,  63, 195, 245, 168,  62,\
                            195, 245, 168,  62, 195, 245, 168,  62, 195, 245,\
                            168,  62, 195, 245, 168,  62, 143, 194,  53,  63 };

    open3d::Image image;

    // test image dimensions
    const int local_width = 5;
    const int local_height = 5;
    const int local_num_of_channels = 1;
    const int local_bytes_per_channel = 4;

    image.PrepareImage(local_width,
                       local_height,
                       local_num_of_channels,
                       local_bytes_per_channel);

    RandInit(image.data_);

    auto outputImage = open3d::CreateFloatImageFromImage(image);

    open3d::ClipIntensityImage(*outputImage, 0.33, 0.71);

    EXPECT_FALSE(outputImage->IsEmpty());
    EXPECT_EQ(local_width, outputImage->width_);
    EXPECT_EQ(local_height, outputImage->height_);
    EXPECT_EQ(local_num_of_channels, outputImage->num_of_channels_);
    EXPECT_EQ(local_bytes_per_channel, outputImage->bytes_per_channel_);
    for (size_t i = 0; i < outputImage->data_.size(); i++)
        EXPECT_EQ(ref[i], outputImage->data_[i]);
}

// ----------------------------------------------------------------------------
// Tests one of the following configurations
// ----------------------------------------------------------------------------
template<typename T>
void TEST_CreateImageFromFloatImage()
{
    open3d::Image image;

    // test image dimensions
    const int local_width = 10;
    const int local_height = 10;
    const int local_num_of_channels = 1;
    const int bytes_per_channel = sizeof(T);

    image.PrepareImage(local_width,
                       local_height,
                       local_num_of_channels,
                       bytes_per_channel);

    RandInit(image.data_);

    auto floatImage = open3d::CreateFloatImageFromImage(image);

    auto outImage = open3d::CreateImageFromFloatImage<T>(*floatImage);

    EXPECT_FALSE(outImage->IsEmpty());
    EXPECT_EQ(local_width, outImage->width_);
    EXPECT_EQ(local_height, outImage->height_);
    EXPECT_EQ(local_num_of_channels, outImage->num_of_channels_);
    EXPECT_EQ(bytes_per_channel, outImage->bytes_per_channel_);
    for (size_t i = 0; i < outImage->data_.size(); i++)
        EXPECT_EQ(image.data_[i], outImage->data_[i]);
}

template void TEST_CreateImageFromFloatImage<uint8_t>();
template void TEST_CreateImageFromFloatImage<uint16_t>();

// ----------------------------------------------------------------------------
// Tests the case output image bytes_per_channel = 1.
// ----------------------------------------------------------------------------
TEST(Image, CreateImageFromFloatImage_8bit)
{
    TEST_CreateImageFromFloatImage<uint8_t>();
}

// ----------------------------------------------------------------------------
// Tests the case output image bytes_per_channel = 2.
// ----------------------------------------------------------------------------
TEST(Image, CreateImageFromFloatImage_16bit)
{
    TEST_CreateImageFromFloatImage<uint16_t>();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, FilterImagePyramid)
{
    // reference data used to validate the filtering of an image
    vector<vector<uint8_t>> ref = {
        { 197, 104,   2, 215, 197, 104, 130, 215, 197, 104,\
            2, 215, 218, 220,  75, 120, 172,  16, 201, 247,\
           40,  29, 165, 249, 197, 104, 130, 215, 197, 104,\
            2, 216, 113,  93, 231, 244,  31,  14, 202, 120,\
          103, 223,  74, 248,  40,  29,  37, 250, 229, 182,\
          200,  88, 207, 183,  11,  89, 113,  93, 103, 245,\
          238, 161,  68, 120, 152,  75, 208, 247,  40,  29,\
          165, 249, 254,   3,  89,  89, 232,   4, 156,  89,\
          113,  93, 231, 244, 113,  93, 103, 245,  34,  20,\
          235, 250,  65,  68, 176, 251,  36,  77, 143, 101,\
          135,  17, 191, 100,  38, 179, 111,  89, 107, 177,\
          239,  88, 180, 222, 106, 251,   7,  39,  48, 252,\
          182, 243,  86, 102,  36,  77, 143, 101, 120,  76,\
          119,  71, 243,  99, 221, 208, 219,  27, 234, 250,\
          228, 148, 175, 251 },
        { 235, 194, 250, 118, 240, 103,  29, 249,  34, 179,\
          255, 249, 187,  86, 117, 118, 222, 197,  23, 250,\
           38,  14, 230, 250,  71, 152,  59, 117, 141,  44,\
          147, 250, 116, 253,  92, 251 }
    };

    open3d::Image image;

    // test image dimensions
    const int local_width = 6;
    const int local_height = 6;
    const int local_num_of_channels = 1;
    const int local_bytes_per_channel = 4;
    const int local_num_of_levels = 2;

    image.PrepareImage(local_width,
                       local_height,
                       local_num_of_channels,
                       local_bytes_per_channel);

    RandInit(image.data_);

    auto floatImage = open3d::CreateFloatImageFromImage(image);

    auto pyramid = open3d::CreateImagePyramid(*floatImage, local_num_of_levels);

    auto outputPyramid = open3d::FilterImagePyramid(pyramid, open3d::Image::FilterType::Gaussian3);

    EXPECT_EQ(pyramid.size(), outputPyramid.size());

    for (size_t p = 0; p < pyramid.size(); p++)
    {
        auto inputImage = pyramid[p];
        auto outputImage = outputPyramid[p];

        EXPECT_FALSE(outputImage->IsEmpty());
        EXPECT_EQ(inputImage->width_, outputImage->width_);
        EXPECT_EQ(inputImage->height_, outputImage->height_);
        EXPECT_EQ(inputImage->num_of_channels_, outputImage->num_of_channels_);
        EXPECT_EQ(inputImage->bytes_per_channel_, outputImage->bytes_per_channel_);
        for (size_t i = 0; i < outputImage->data_.size(); i++)
            EXPECT_EQ(ref[p][i], outputImage->data_[i]);
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, CreateImagePyramid)
{
    // reference data used to validate the filtering of an image
    vector<vector<uint8_t>> ref = {
        {  70, 248,  76, 242, 253,  56, 240, 133,  74, 177,\
          235, 106,  44,  98, 118, 118,  65,  41,   9, 236,\
           25,  56,  17,  84,  95, 126, 236, 213,  96,   8,\
           79, 166,   2, 155, 154,   0, 211, 140, 134,  31,\
           62, 114, 137, 106, 213,   1, 224,  23,  43, 233,\
            5,  68,  35,  22, 153, 131, 149, 134,  89, 245,\
          143, 168, 157, 145,  69,  56, 146,  26, 196,  25,\
           57,   3, 140, 195, 110,  98, 196,  80, 122, 239,\
           58, 127,  53,  94, 150, 206, 225,  44,  85,  60,\
           34, 228, 228, 191, 119,  42, 248,  11,  69, 190,\
           36, 126, 193, 177,  66,  49,  21,   7, 129, 143,\
          247, 188,  16,  45,  27, 167, 251, 252, 211,  82,\
           57, 246,  56,  31, 183, 175,  74, 177, 186, 143,\
          112, 223,  15,  51, 146,  81, 100, 167,  89, 229,\
           56,  81, 162,  73 },
        {  69,  98, 134, 242,  81, 151,  46, 117,  21,  98,\
          118, 116,  35,  72,  35, 243,  69, 189, 244, 243,\
          140,  20,  78, 104, 147, 188,  59, 237, 253,  80,\
          122, 236,  95, 216,  42, 228 }
    };

    open3d::Image image;

    // test image dimensions
    const int local_width = 6;
    const int local_height = 6;
    const int local_num_of_channels = 1;
    const int local_bytes_per_channel = 4;
    const int local_num_of_levels = 2;

    image.PrepareImage(local_width,
                       local_height,
                       local_num_of_channels,
                       local_bytes_per_channel);

    RandInit(image.data_);

    auto floatImage = open3d::CreateFloatImageFromImage(image);

    auto pyramid = open3d::CreateImagePyramid(*floatImage, local_num_of_levels);

    int expected_width = local_width;
    int expected_height = local_width;
    for (size_t p = 0; p < pyramid.size(); p++)
    {
        auto outputImage = pyramid[p];

        EXPECT_FALSE(outputImage->IsEmpty());
        EXPECT_EQ(expected_width, outputImage->width_);
        EXPECT_EQ(expected_height, outputImage->height_);
        EXPECT_EQ(local_num_of_channels, outputImage->num_of_channels_);
        EXPECT_EQ(local_bytes_per_channel, outputImage->bytes_per_channel_);
        for (size_t i = 0; i < outputImage->data_.size(); i++)
            EXPECT_EQ(ref[p][i], outputImage->data_[i]);

        expected_width /= 2;
        expected_height /= 2;
    }
}
