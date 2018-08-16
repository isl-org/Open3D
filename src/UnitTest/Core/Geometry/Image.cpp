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
// Tests one of the following configurations
// channels: bytes per channel
// 1: 1/2/4
// 3: 1/2/4 with either Equal or Weighted type
// ----------------------------------------------------------------------------
void CreateFloatImageFromImage(const int& num_of_channels,
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

    randInit(image.data_);

    auto floatImage = open3d::CreateFloatImageFromImage(image);

    // display float image data
    // for (size_t i = 0; i < floatImage->data_.size(); i++)
    //     {
    //         if ((i % 10 == 0) && (i != 0))
    //             cout << "\\" << endl;
    //         cout << setw(4) << (float)floatImage->data_[i] << ",";
    //     }
    // cout << endl;

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

    CreateFloatImageFromImage(1, 1, ref, open3d::Image::ColorToIntensityConversionType::Weighted);
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

    CreateFloatImageFromImage(1, 2, ref, open3d::Image::ColorToIntensityConversionType::Weighted);
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

    CreateFloatImageFromImage(1, 4, ref, open3d::Image::ColorToIntensityConversionType::Weighted);
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

    CreateFloatImageFromImage(3, 1, ref, open3d::Image::ColorToIntensityConversionType::Weighted);
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

    CreateFloatImageFromImage(3, 1, ref, open3d::Image::ColorToIntensityConversionType::Equal);
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

    CreateFloatImageFromImage(3, 2, ref, open3d::Image::ColorToIntensityConversionType::Weighted);
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

    CreateFloatImageFromImage(3, 2, ref, open3d::Image::ColorToIntensityConversionType::Equal);
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

    CreateFloatImageFromImage(3, 4, ref, open3d::Image::ColorToIntensityConversionType::Weighted);
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

    CreateFloatImageFromImage(3, 4, ref, open3d::Image::ColorToIntensityConversionType::Equal);
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

    randInit(image.data_);

    auto floatImage = open3d::ConvertDepthToFloatImage(image);

    // display float image data
    // for (size_t i = 0; i < floatImage->data_.size(); i++)
    //     {
    //         if ((i % 10 == 0) && (i != 0))
    //             cout << "\\" << endl;
    //         cout << setw(4) << (float)floatImage->data_[i] << ",";
    //     }
    // cout << endl;

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
// Tests one of the following configurations
// ----------------------------------------------------------------------------
template<typename T>
void CreateImageFromFloatImage()
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

    randInit(image.data_);

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

template void CreateImageFromFloatImage<uint8_t>();
template void CreateImageFromFloatImage<uint16_t>();

// ----------------------------------------------------------------------------
// Tests the case output image bytes_per_channel = 1.
// ----------------------------------------------------------------------------
TEST(Image, CreateImageFromFloatImage_8bit)
{
    CreateImageFromFloatImage<uint8_t>();
}

// ----------------------------------------------------------------------------
// Tests the case output image bytes_per_channel = 2.
// ----------------------------------------------------------------------------
TEST(Image, CreateImageFromFloatImage_16bit)
{
    CreateImageFromFloatImage<uint16_t>();
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