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

#include "Open3D/Geometry/Image.h"
#include "Open3D/Camera/PinholeCameraIntrinsic.h"
#include "TestUtility/UnitTest.h"

using namespace Eigen;
using namespace open3d;
using namespace std;
using namespace unit_test;

using ConversionType = geometry::Image::ColorToIntensityConversionType;
using FilterType = geometry::Image::FilterType;

// ----------------------------------------------------------------------------
// test the default constructor scenario
// ----------------------------------------------------------------------------
TEST(Image, DefaultConstructor) {
    geometry::Image image;

    // inherited from Geometry2D
    EXPECT_EQ(geometry::Geometry::GeometryType::Image, image.GetGeometryType());
    EXPECT_EQ(2, image.Dimension());

    // public member variables
    EXPECT_EQ(0, image.width_);
    EXPECT_EQ(0, image.height_);
    EXPECT_EQ(0, image.num_of_channels_);
    EXPECT_EQ(0, image.bytes_per_channel_);
    EXPECT_EQ(0u, image.data_.size());

    // public members
    EXPECT_TRUE(image.IsEmpty());
    EXPECT_FALSE(image.HasData());

    ExpectEQ(Zero2d, image.GetMinBound());
    ExpectEQ(Zero2d, image.GetMaxBound());

    EXPECT_FALSE(image.TestImageBoundary(0, 0));
    EXPECT_EQ(0, image.BytesPerLine());
}

// ----------------------------------------------------------------------------
// test Prepare aka image creation
// ----------------------------------------------------------------------------
TEST(Image, CreateImage) {
    int width = 1920;
    int height = 1080;
    int num_of_channels = 3;
    int bytes_per_channel = 1;

    geometry::Image image;

    image.Prepare(width, height, num_of_channels, bytes_per_channel);

    // public member variables
    EXPECT_EQ(width, image.width_);
    EXPECT_EQ(height, image.height_);
    EXPECT_EQ(num_of_channels, image.num_of_channels_);
    EXPECT_EQ(bytes_per_channel, image.bytes_per_channel_);
    EXPECT_EQ(size_t(width * height * num_of_channels * bytes_per_channel),
              image.data_.size());

    // public members
    EXPECT_FALSE(image.IsEmpty());
    EXPECT_TRUE(image.HasData());

    ExpectEQ(Zero2d, image.GetMinBound());
    ExpectEQ(Vector2d(width, height), image.GetMaxBound());

    EXPECT_TRUE(image.TestImageBoundary(0, 0));
    EXPECT_EQ(width * num_of_channels * bytes_per_channel,
              image.BytesPerLine());
}

// ----------------------------------------------------------------------------
// test Clear
// ----------------------------------------------------------------------------
TEST(Image, Clear) {
    int width = 1920;
    int height = 1080;
    int num_of_channels = 3;
    int bytes_per_channel = 1;

    geometry::Image image;

    image.Prepare(width, height, num_of_channels, bytes_per_channel);

    image.Clear();

    // public member variables
    EXPECT_EQ(0, image.width_);
    EXPECT_EQ(0, image.height_);
    EXPECT_EQ(0, image.num_of_channels_);
    EXPECT_EQ(0, image.bytes_per_channel_);
    EXPECT_EQ(0u, image.data_.size());

    // public members
    EXPECT_TRUE(image.IsEmpty());
    EXPECT_FALSE(image.HasData());

    ExpectEQ(Zero2d, image.GetMinBound());
    ExpectEQ(Zero2d, image.GetMaxBound());

    EXPECT_FALSE(image.TestImageBoundary(0, 0));
    EXPECT_EQ(0, image.BytesPerLine());
}

// ----------------------------------------------------------------------------
// test FloatValueAt, bilinear(?) interpolation
// ----------------------------------------------------------------------------
TEST(Image, FloatValueAt) {
    geometry::Image image;

    int width = 10;
    int height = 10;
    int num_of_channels = 1;
    int bytes_per_channel = 4;

    image.Prepare(width, height, num_of_channels, bytes_per_channel);

    float* const im = Cast<float>(&image.data_[0]);

    im[0 * width + 0] = 4.0f;
    im[0 * width + 1] = 4.0f;
    im[1 * width + 0] = 4.0f;
    im[1 * width + 1] = 4.0f;

    EXPECT_NEAR(4.0f, image.FloatValueAt(0.0, 0.0).second, THRESHOLD_1E_6);
    EXPECT_NEAR(4.0f, image.FloatValueAt(0.0, 1.0).second, THRESHOLD_1E_6);
    EXPECT_NEAR(4.0f, image.FloatValueAt(1.0, 0.0).second, THRESHOLD_1E_6);
    EXPECT_NEAR(4.0f, image.FloatValueAt(1.0, 1.0).second, THRESHOLD_1E_6);
    EXPECT_NEAR(4.0f, image.FloatValueAt(0.5, 0.5).second, THRESHOLD_1E_6);
    EXPECT_NEAR(2.0f, image.FloatValueAt(0.0, 1.5).second, THRESHOLD_1E_6);
    EXPECT_NEAR(2.0f, image.FloatValueAt(1.5, 0.0).second, THRESHOLD_1E_6);
    EXPECT_NEAR(1.0f, image.FloatValueAt(1.5, 1.5).second, THRESHOLD_1E_6);
}

// ----------------------------------------------------------------------------
// member data is not private and as such can lead to errors
// ----------------------------------------------------------------------------
TEST(Image, DISABLED_MemberData) {
    int width = 1920;
    int height = 1080;
    int num_of_channels = 3;
    int bytes_per_channel = 1;

    geometry::Image image;

    image.Prepare(width, height, num_of_channels, bytes_per_channel);

    int temp_width = 320;
    int temp_height = 240;
    int temp_num_of_channels = 1;
    int temp_bytes_per_channel = 3;

    image.width_ = temp_width;
    EXPECT_EQ(temp_width * height * num_of_channels * bytes_per_channel,
              int(image.data_.size()));

    image.width_ = width;
    image.height_ = temp_height;
    EXPECT_EQ(width * temp_height * num_of_channels * bytes_per_channel,
              int(image.data_.size()));

    image.height_ = height;
    image.num_of_channels_ = temp_num_of_channels;
    EXPECT_EQ(width * height * temp_num_of_channels * bytes_per_channel,
              int(image.data_.size()));

    image.num_of_channels_ = num_of_channels;
    image.bytes_per_channel_ = temp_bytes_per_channel;
    EXPECT_EQ(width * height * num_of_channels * temp_bytes_per_channel,
              int(image.data_.size()));

    image.bytes_per_channel_ = bytes_per_channel;
    image.data_ = vector<uint8_t>();
    EXPECT_EQ(width * height * num_of_channels * bytes_per_channel,
              int(image.data_.size()));
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, CreateDepthToCameraDistanceMultiplierFloatImage) {
    camera::PinholeCameraIntrinsic intrinsic = camera::PinholeCameraIntrinsic(
            camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);

    auto image =
            geometry::Image::CreateDepthToCameraDistanceMultiplierFloatImage(
                    intrinsic);

    // test image dimensions
    int width = 640;
    int height = 480;
    int num_of_channels = 1;
    int bytes_per_channel = 4;

    EXPECT_FALSE(image->IsEmpty());
    EXPECT_EQ(width, image->width_);
    EXPECT_EQ(height, image->height_);
    EXPECT_EQ(num_of_channels, image->num_of_channels_);
    EXPECT_EQ(bytes_per_channel, image->bytes_per_channel_);
}

// ----------------------------------------------------------------------------
// Tests one of the following configurations
// channels: bytes per channel
// 1: 1/2/4
// 3: 1/2/4 with either Equal or Weighted type
// ----------------------------------------------------------------------------
void TEST_CreateFloatImage(
        const int& num_of_channels,
        const int& bytes_per_channel,
        const vector<uint8_t>& ref,
        const geometry::Image::ColorToIntensityConversionType& type) {
    geometry::Image image;

    // test image dimensions
    int width = 5;
    int height = 5;
    int float_num_of_channels = 1;

    image.Prepare(width, height, num_of_channels, bytes_per_channel);

    Rand(image.data_, 0, 255, 0);

    auto float_image = image.CreateFloatImage();

    EXPECT_FALSE(float_image->IsEmpty());
    EXPECT_EQ(width, float_image->width_);
    EXPECT_EQ(height, float_image->height_);
    EXPECT_EQ(float_num_of_channels, float_image->num_of_channels_);
    EXPECT_EQ(int(sizeof(float)), float_image->bytes_per_channel_);
    ExpectEQ(ref, float_image->data_);
}

// ----------------------------------------------------------------------------
// Tests one of the following configuration:
// channels: 1
// bytes per channel: 1
// ----------------------------------------------------------------------------
TEST(Image, CreateFloatImage_1_1) {
    // reference data used to validate the creation of the float image
    vector<uint8_t> ref = {
            215, 214, 86,  63,  201, 200, 200, 62,  200, 199, 71,  63,  204,
            203, 75,  63,  233, 232, 104, 63,  201, 200, 72,  62,  171, 170,
            170, 62,  196, 195, 67,  63,  141, 140, 140, 62,  142, 141, 13,
            63,  243, 242, 242, 62,  161, 160, 32,  63,  187, 186, 186, 62,
            131, 130, 2,   63,  243, 242, 114, 63,  234, 233, 105, 63,  163,
            162, 34,  63,  183, 182, 54,  63,  145, 144, 16,  62,  155, 154,
            26,  63,  129, 128, 128, 60,  245, 244, 116, 62,  137, 136, 8,
            62,  206, 205, 77,  63,  157, 156, 28,  62};

    TEST_CreateFloatImage(1, 1, ref, ConversionType::Weighted);
}

// ----------------------------------------------------------------------------
// Tests one of the following configuration:
// channels: 1
// bytes per channel: 2
// ----------------------------------------------------------------------------
TEST(Image, CreateFloatImage_1_2) {
    // reference data used to validate the creation of the float image
    vector<uint8_t> ref = {
            0, 172, 201, 70, 0, 199, 75,  71, 0, 160, 75,  70, 0, 85,  67,  71,
            0, 70,  13,  71, 0, 121, 32,  71, 0, 93,  2,   71, 0, 242, 105, 71,
            0, 162, 54,  71, 0, 36,  26,  71, 0, 16,  116, 70, 0, 34,  77,  71,
            0, 78,  204, 70, 0, 8,   217, 69, 0, 248, 95,  70, 0, 130, 85,  71,
            0, 56,  151, 70, 0, 162, 5,   71, 0, 125, 120, 71, 0, 74,  68,  71,
            0, 134, 68,  71, 0, 102, 99,  71, 0, 144, 178, 70, 0, 205, 106, 71,
            0, 17,  114, 71};

    TEST_CreateFloatImage(1, 2, ref, ConversionType::Weighted);
}

// ----------------------------------------------------------------------------
// Tests one of the following configuration:
// channels: 1
// bytes per channel: 4
// ----------------------------------------------------------------------------
TEST(Image, CreateFloatImage_1_4) {
    // reference data used to validate the creation of the float image
    vector<uint8_t> ref = {
            214, 100, 199, 203, 232, 50,  85,  195, 70,  141, 121, 160, 93,
            130, 242, 233, 162, 182, 36,  154, 4,   61,  34,  205, 39,  102,
            33,  27,  254, 55,  130, 213, 156, 75,  162, 133, 125, 248, 74,
            196, 134, 196, 102, 227, 72,  89,  205, 234, 17,  242, 134, 21,
            49,  169, 227, 88,  16,  5,   116, 16,  60,  247, 230, 216, 67,
            137, 95,  193, 130, 170, 135, 10,  111, 237, 237, 183, 72,  188,
            163, 90,  175, 42,  112, 224, 211, 84,  58,  227, 89,  175, 243,
            150, 167, 218, 112, 235, 101, 207, 174, 232};

    TEST_CreateFloatImage(1, 4, ref, ConversionType::Weighted);
}

// ----------------------------------------------------------------------------
// Tests one of the following configuration:
// channels: 3
// bytes per channel: 1
// ColorToIntensityConversionType: Weighted
// ----------------------------------------------------------------------------
TEST(Image, CreateFloatImage_3_1_Weighted) {
    // reference data used to validate the creation of the float image
    vector<uint8_t> ref = {
            45,  241, 17,  63,  29,  96,  75,  63,  154, 112, 20,  63,  0,
            241, 3,   63,  180, 56,  4,   63,  139, 60,  58,  63,  115, 8,
            204, 62,  216, 59,  119, 62,  64,  47,  151, 62,  251, 20,  36,
            63,  194, 101, 54,  63,  138, 51,  5,   63,  54,  35,  64,  63,
            94,  59,  32,  63,  29,  161, 44,  63,  137, 77,  46,  63,  199,
            12,  35,  63,  121, 21,  90,  62,  101, 168, 243, 62,  209, 97,
            143, 62,  9,   228, 61,  63,  224, 255, 239, 62,  57,  33,  29,
            63,  197, 186, 3,   63,  145, 27,  72,  63};

    TEST_CreateFloatImage(3, 1, ref, ConversionType::Weighted);
}

// ----------------------------------------------------------------------------
// Tests one of the following configuration:
// channels: 3
// bytes per channel: 1
// ColorToIntensityConversionType: Equal
// ----------------------------------------------------------------------------
TEST(Image, CreateFloatImage_3_1_Equal) {
    // reference data used to validate the creation of the float image
    vector<uint8_t> ref = {
            45,  241, 17,  63,  29,  96,  75,  63,  154, 112, 20,  63,  0,
            241, 3,   63,  180, 56,  4,   63,  139, 60,  58,  63,  115, 8,
            204, 62,  216, 59,  119, 62,  64,  47,  151, 62,  251, 20,  36,
            63,  194, 101, 54,  63,  138, 51,  5,   63,  54,  35,  64,  63,
            94,  59,  32,  63,  29,  161, 44,  63,  137, 77,  46,  63,  199,
            12,  35,  63,  121, 21,  90,  62,  101, 168, 243, 62,  209, 97,
            143, 62,  9,   228, 61,  63,  224, 255, 239, 62,  57,  33,  29,
            63,  197, 186, 3,   63,  145, 27,  72,  63};

    TEST_CreateFloatImage(3, 1, ref, ConversionType::Equal);
}

// ----------------------------------------------------------------------------
// Tests one of the following configuration:
// channels: 3
// bytes per channel: 2
// ColorToIntensityConversionType: Weighted
// ----------------------------------------------------------------------------
TEST(Image, CreateFloatImage_3_2_Weighted) {
    // reference data used to validate the creation of the float image
    vector<uint8_t> ref = {
            16,  146, 27,  71,  44,  160, 31,  71,  234, 31,  69,  71,  39,
            148, 210, 70,  195, 103, 83,  70,  79,  233, 246, 70,  97,  236,
            83,  71,  226, 42,  19,  71,  145, 153, 208, 70,  82,  101, 251,
            69,  235, 227, 88,  71,  45,  27,  31,  71,  209, 107, 72,  71,
            169, 123, 155, 70,  236, 187, 50,  71,  151, 82,  72,  71,  48,
            235, 76,  71,  32,  111, 86,  71,  27,  105, 148, 70,  71,  196,
            219, 70,  12,  108, 22,  71,  198, 41,  183, 70,  225, 5,   23,
            71,  210, 181, 85,  71,  101, 14,  28,  71};

    TEST_CreateFloatImage(3, 2, ref, ConversionType::Weighted);
}

// ----------------------------------------------------------------------------
// Tests one of the following configuration:
// channels: 3
// bytes per channel: 2
// ColorToIntensityConversionType: Equal
// ----------------------------------------------------------------------------
TEST(Image, CreateFloatImage_3_2_Equal) {
    // reference data used to validate the creation of the float image
    vector<uint8_t> ref = {
            16,  146, 27,  71,  44,  160, 31,  71,  234, 31,  69,  71,  39,
            148, 210, 70,  195, 103, 83,  70,  79,  233, 246, 70,  97,  236,
            83,  71,  226, 42,  19,  71,  145, 153, 208, 70,  82,  101, 251,
            69,  235, 227, 88,  71,  45,  27,  31,  71,  209, 107, 72,  71,
            169, 123, 155, 70,  236, 187, 50,  71,  151, 82,  72,  71,  48,
            235, 76,  71,  32,  111, 86,  71,  27,  105, 148, 70,  71,  196,
            219, 70,  12,  108, 22,  71,  198, 41,  183, 70,  225, 5,   23,
            71,  210, 181, 85,  71,  101, 14,  28,  71};

    TEST_CreateFloatImage(3, 2, ref, ConversionType::Equal);
}

// ----------------------------------------------------------------------------
// Tests one of the following configuration:
// channels: 3
// bytes per channel: 4
// ColorToIntensityConversionType: Weighted
// ----------------------------------------------------------------------------
TEST(Image, CreateFloatImage_3_4_Weighted) {
    // reference data used to validate the creation of the float image
    vector<uint8_t> ref = {
            153, 122, 238, 202, 65,  5,   17,  233, 117, 224, 24,  213, 166,
            79,  59,  233, 15,  163, 133, 88,  22,  30,  10,  216, 24,  168,
            218, 222, 111, 170, 219, 233, 198, 232, 16,  109, 227, 84,  156,
            229, 56,  95,  77,  97,  226, 226, 200, 188, 36,  128, 64,  193,
            178, 161, 146, 208, 240, 239, 83,  208, 189, 119, 176, 114, 209,
            111, 82,  249, 14,  45,  72,  210, 222, 97,  25,  247, 179, 223,
            15,  114, 245, 201, 149, 76,  224, 3,   24,  64,  17,  103, 98,
            222, 145, 236, 94,  233, 36,  85,  141, 233};

    TEST_CreateFloatImage(3, 4, ref, ConversionType::Weighted);
}

// ----------------------------------------------------------------------------
// Tests one of the following configuration:
// channels: 3
// bytes per channel: 4
// ColorToIntensityConversionType: Equal
// ----------------------------------------------------------------------------
TEST(Image, CreateFloatImage_3_4_Equal) {
    // reference data used to validate the creation of the float image
    vector<uint8_t> ref = {
            153, 122, 238, 202, 65,  5,   17,  233, 117, 224, 24,  213, 166,
            79,  59,  233, 15,  163, 133, 88,  22,  30,  10,  216, 24,  168,
            218, 222, 111, 170, 219, 233, 198, 232, 16,  109, 227, 84,  156,
            229, 56,  95,  77,  97,  226, 226, 200, 188, 36,  128, 64,  193,
            178, 161, 146, 208, 240, 239, 83,  208, 189, 119, 176, 114, 209,
            111, 82,  249, 14,  45,  72,  210, 222, 97,  25,  247, 179, 223,
            15,  114, 245, 201, 149, 76,  224, 3,   24,  64,  17,  103, 98,
            222, 145, 236, 94,  233, 36,  85,  141, 233};

    TEST_CreateFloatImage(3, 4, ref, ConversionType::Equal);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, PointerAt) {
    geometry::Image image;

    int width = 10;
    int height = 10;
    int num_of_channels = 1;
    int bytes_per_channel = 4;

    image.Prepare(width, height, num_of_channels, bytes_per_channel);

    float* const im = Cast<float>(&image.data_[0]);

    im[0 * width + 0] = 0.0f;
    im[0 * width + 1] = 1.0f;
    im[1 * width + 0] = 2.0f;
    im[1 * width + 1] = 3.0f;

    EXPECT_NEAR(0.0f, *image.PointerAt<float>(0, 0), THRESHOLD_1E_6);
    EXPECT_NEAR(1.0f, *image.PointerAt<float>(1, 0), THRESHOLD_1E_6);
    EXPECT_NEAR(2.0f, *image.PointerAt<float>(0, 1), THRESHOLD_1E_6);
    EXPECT_NEAR(3.0f, *image.PointerAt<float>(1, 1), THRESHOLD_1E_6);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, ConvertDepthToFloatImage) {
    // reference data used to validate the creation of the float image
    vector<uint8_t> ref = {
            208, 254, 91,  58,  103, 154, 205, 57,  59,  147, 76,  58,  236,
            175, 80,  58,  232, 127, 110, 58,  103, 154, 77,  57,  62,  195,
            174, 57,  139, 118, 72,  58,  22,  236, 143, 57,  66,  243, 16,
            58,  161, 199, 248, 57,  134, 123, 36,  58,  255, 53,  191, 57,
            93,  164, 5,   58,  161, 199, 120, 58,  20,  135, 111, 58,  222,
            137, 38,  58,  79,  25,  59,  58,  198, 8,   20,  57,  126, 80,
            30,  58,  5,   150, 131, 55,  249, 213, 122, 57,  101, 207, 11,
            57,  68,  190, 82,  58,  214, 94,  32,  57};

    geometry::Image image;

    // test image dimensions
    int width = 5;
    int height = 5;
    int num_of_channels = 1;
    int bytes_per_channel = 1;
    int float_num_of_channels = 1;

    image.Prepare(width, height, num_of_channels, bytes_per_channel);

    Rand(image.data_, 0, 255, 0);

    auto float_image = image.ConvertDepthToFloatImage();

    EXPECT_FALSE(float_image->IsEmpty());
    EXPECT_EQ(width, float_image->width_);
    EXPECT_EQ(height, float_image->height_);
    EXPECT_EQ(float_num_of_channels, float_image->num_of_channels_);
    EXPECT_EQ(int(sizeof(float)), float_image->bytes_per_channel_);
    ExpectEQ(ref, float_image->data_);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, FlipImage) {
    // reference data used to validate the creation of the float image
    vector<uint8_t> ref = {
            233, 45,  204, 198, 205, 80,  90,  190, 133, 138, 127, 155, 87,
            84,  248, 228, 162, 170, 40,  149, 207, 33,  38,  200, 202, 69,
            37,  22,  14,  88,  133, 208, 193, 48,  166, 128, 138, 215, 79,
            191, 92,  78,  108, 222, 242, 70,  210, 229, 44,  47,  138, 16,
            0,   0,   0,   0,   82,  224, 121, 11,  74,  130, 236, 211, 171,
            230, 100, 188, 10,  236, 138, 5,   67,  163, 243, 178, 0,   0,
            0,   0,   69,  238, 117, 219, 165, 205, 62,  222, 140, 136, 249,
            145, 118, 162, 118, 230, 110, 1,   179, 227};

    geometry::Image image;

    // test image dimensions
    int width = 5;
    int height = 5;
    int num_of_channels = 1;
    int bytes_per_channel = 4;
    int flip_bytes_per_channel = 1;

    image.Prepare(width, height, num_of_channels, bytes_per_channel);

    Rand(image.data_, 0, 255, 0);

    auto flip_image = image.ConvertDepthToFloatImage();

    EXPECT_FALSE(flip_image->IsEmpty());
    EXPECT_EQ(width, flip_image->width_);
    EXPECT_EQ(height, flip_image->height_);
    EXPECT_EQ(flip_bytes_per_channel, flip_image->num_of_channels_);
    EXPECT_EQ(int(sizeof(float)), flip_image->bytes_per_channel_);
    ExpectEQ(ref, flip_image->data_);
}

// ----------------------------------------------------------------------------
// Tests one of the following configurations
// channels: bytes per channel
// 1: 1/2/4
// 3: 1/2/4 with either Equal or Weighted type
// ----------------------------------------------------------------------------
void TEST_Filter(const vector<uint8_t>& ref,
                 const geometry::Image::FilterType& filter) {
    geometry::Image image;

    // test image dimensions
    int width = 5;
    int height = 5;
    int num_of_channels = 1;
    int bytes_per_channel = 4;

    image.Prepare(width, height, num_of_channels, bytes_per_channel);

    Rand(image.data_, 0, 255, 0);

    auto float_image = image.CreateFloatImage();

    auto output = float_image->Filter(filter);

    EXPECT_FALSE(output->IsEmpty());
    EXPECT_EQ(width, output->width_);
    EXPECT_EQ(height, output->height_);
    EXPECT_EQ(num_of_channels, output->num_of_channels_);
    EXPECT_EQ(bytes_per_channel, output->bytes_per_channel_);
    ExpectEQ(ref, output->data_);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, Filter_Gaussian3) {
    // reference data used to validate the filtering of an image
    vector<uint8_t> ref = {
            41,  194, 49,  204, 116, 56,  130, 211, 198, 225, 181, 232, 198,
            225, 53,  233, 198, 225, 181, 232, 177, 94,  205, 232, 47,  90,
            77,  233, 240, 252, 4,   233, 93,  130, 114, 232, 93,  130, 242,
            231, 177, 94,  77,  233, 47,  90,  205, 233, 72,  89,  77,  233,
            6,   134, 220, 88,  128, 234, 129, 89,  60,  96,  205, 232, 167,
            91,  77,  233, 2,   196, 171, 233, 229, 149, 243, 233, 12,  159,
            128, 233, 36,  49,  20,  226, 223, 39,  141, 226, 137, 164, 52,
            234, 108, 176, 182, 234, 146, 238, 64,  234};

    TEST_Filter(ref, FilterType::Gaussian3);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, Filter_Gaussian5) {
    // reference data used to validate the filtering of an image
    vector<uint8_t> ref = {
            61,  94,  205, 231, 230, 96,  109, 232, 15,  16,  218, 232, 2,
            118, 3,   233, 160, 185, 166, 232, 61,  94,  205, 232, 46,  125,
            35,  233, 60,  145, 12,  233, 110, 3,   165, 232, 122, 145, 23,
            232, 223, 6,   26,  233, 23,  249, 119, 233, 159, 37,  94,  233,
            234, 229, 13,  233, 99,  24,  143, 232, 41,  96,  205, 232, 206,
            73,  101, 233, 15,  186, 202, 233, 62,  231, 242, 233, 76,  236,
            159, 233, 35,  111, 205, 231, 102, 26,  76,  233, 255, 241, 44,
            234, 32,  174, 126, 234, 84,  234, 47,  234};

    TEST_Filter(ref, FilterType::Gaussian5);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, Filter_Gaussian7) {
    // reference data used to validate the filtering of an image
    vector<uint8_t> ref = {
            71,  19,  68,  232, 29,  11,  169, 232, 178, 140, 214, 232, 35,
            21,  214, 232, 245, 42,  147, 232, 66,  168, 175, 232, 125, 101,
            5,   233, 242, 119, 15,  233, 60,  92,  246, 232, 131, 231, 154,
            232, 226, 75,  240, 232, 83,  18,  69,  233, 128, 68,  108, 233,
            67,  141, 98,  233, 63,  199, 27,  233, 108, 191, 244, 232, 122,
            49,  127, 233, 20,  166, 194, 233, 176, 46,  222, 233, 32,  207,
            168, 233, 187, 237, 232, 232, 99,  40,  161, 233, 128, 206, 18,
            234, 108, 135, 55,  234, 187, 97,  17,  234};

    TEST_Filter(ref, FilterType::Gaussian7);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, Filter_Sobel3Dx) {
    // reference data used to validate the filtering of an image
    vector<uint8_t> ref = {
            172, 2,   109, 77,  136, 55,  130, 213, 198, 225, 181, 234, 254,
            55,  130, 85,  198, 225, 181, 106, 122, 87,  205, 234, 134, 196,
            102, 99,  177, 184, 144, 106, 254, 55,  2,   86,  93,  130, 242,
            105, 122, 87,  77,  235, 138, 196, 230, 99,  72,  89,  77,  107,
            214, 220, 163, 90,  34,  71,  135, 90,  231, 88,  205, 234, 63,
            133, 106, 99,  73,  45,  10,  235, 101, 207, 174, 232, 44,  100,
            107, 107, 28,  239, 8,   228, 119, 32,  52,  97,  114, 163, 52,
            236, 140, 27,  131, 233, 33,  139, 48,  108};

    TEST_Filter(ref, FilterType::Sobel3Dx);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, Filter_Sobel3Dy) {
    // reference data used to validate the filtering of an image
    vector<uint8_t> ref = {
            151, 248, 205, 205, 67,  56,  130, 213, 93,  130, 242, 105, 93,
            130, 114, 106, 93,  130, 242, 105, 177, 94,  205, 234, 47,  90,
            77,  235, 177, 184, 144, 234, 93,  130, 114, 106, 93,  130, 242,
            105, 108, 57,  173, 217, 91,  238, 228, 216, 254, 55,  2,   86,
            214, 220, 163, 90,  108, 154, 117, 91,  38,  93,  205, 106, 183,
            88,  77,  107, 189, 46,  10,  235, 229, 149, 243, 235, 12,  159,
            128, 235, 189, 150, 69,  227, 36,  53,  188, 227, 97,  219, 112,
            235, 229, 149, 243, 235, 12,  159, 128, 235};

    TEST_Filter(ref, FilterType::Sobel3Dy);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, FilterHorizontal) {
    // reference data used to validate the filtering of an image
    vector<uint8_t> ref = {
            187, 139, 149, 203, 171, 101, 199, 202, 93,  130, 242, 232, 93,
            130, 114, 233, 93,  130, 242, 232, 134, 91,  243, 204, 79,  56,
            130, 212, 254, 55,  2,   213, 254, 55,  130, 212, 94,  58,  24,
            196, 177, 94,  205, 233, 47,  90,  77,  234, 72,  89,  205, 233,
            49,  169, 99,  88,  49,  169, 227, 87,  109, 57,  173, 216, 60,
            247, 230, 215, 97,  137, 95,  192, 72,  188, 163, 89,  108, 154,
            117, 90,  211, 150, 69,  226, 40,  53,  188, 226, 97,  219, 112,
            234, 229, 149, 243, 234, 12,  159, 128, 234};

    geometry::Image image;

    // test image dimensions
    int width = 5;
    int height = 5;
    int num_of_channels = 1;
    int bytes_per_channel = 4;

    image.Prepare(width, height, num_of_channels, bytes_per_channel);

    Rand(image.data_, 0, 255, 0);

    auto float_image = image.CreateFloatImage();

    const std::vector<double> Gaussian3 = {0.25, 0.5, 0.25};

    auto output = float_image->FilterHorizontal(Gaussian3);

    EXPECT_FALSE(output->IsEmpty());
    EXPECT_EQ(width, output->width_);
    EXPECT_EQ(height, output->height_);
    EXPECT_EQ(num_of_channels, output->num_of_channels_);
    EXPECT_EQ(bytes_per_channel, output->bytes_per_channel_);
    ExpectEQ(ref, output->data_);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, Downsample) {
    // reference data used to validate the filtering of an image
    vector<uint8_t> ref = {172, 41, 59,  204, 93, 130, 242, 232,
                           22,  91, 205, 233, 49, 169, 227, 87};

    geometry::Image image;

    // test image dimensions
    int width = 5;
    int height = 5;
    int num_of_channels = 1;
    int bytes_per_channel = 4;

    image.Prepare(width, height, num_of_channels, bytes_per_channel);

    Rand(image.data_, 0, 255, 0);

    auto float_image = image.CreateFloatImage();

    auto output = float_image->Downsample();

    EXPECT_FALSE(output->IsEmpty());
    EXPECT_EQ((int)(width / 2), output->width_);
    EXPECT_EQ((int)(height / 2), output->height_);
    EXPECT_EQ(num_of_channels, output->num_of_channels_);
    EXPECT_EQ(bytes_per_channel, output->bytes_per_channel_);
    ExpectEQ(ref, output->data_);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, Dilate) {
    // reference data used to validate the filtering of an image
    vector<uint8_t> ref = {
            255, 255, 0,   0,   0,   0,   0,   255, 255, 255, 255, 255, 0,
            0,   0,   0,   255, 255, 255, 255, 0,   0,   0,   0,   0,   255,
            255, 255, 255, 255, 0,   0,   0,   0,   255, 255, 255, 255, 255,
            0,   0,   0,   0,   255, 255, 255, 255, 255, 0,   0,   0,   0,
            255, 255, 255, 255, 255, 0,   0,   0,   0,   255, 255, 255, 255,
            255, 0,   0,   0,   0,   255, 255, 255, 255, 255, 0,   0,   0,
            0,   0,   255, 255, 255, 255, 0,   0,   0,   0,   255, 255, 255,
            255, 255, 0,   0,   0,   0,   0,   255, 255};

    geometry::Image image;

    // test image dimensions
    int width = 10;
    int height = 10;
    int num_of_channels = 1;
    int bytes_per_channel = 1;

    image.Prepare(width, height, num_of_channels, bytes_per_channel);

    Rand(image.data_, 0, 255, 0);
    for (size_t i = 0; i < image.data_.size(); i++)
        if (i % 9 == 0) image.data_[i] = 255;

    auto output = image.Dilate();

    EXPECT_FALSE(output->IsEmpty());
    EXPECT_EQ(width, output->width_);
    EXPECT_EQ(height, output->height_);
    EXPECT_EQ(num_of_channels, output->num_of_channels_);
    EXPECT_EQ(bytes_per_channel, output->bytes_per_channel_);
    ExpectEQ(ref, output->data_);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, LinearTransform) {
    // reference data used to validate the filtering of an image
    vector<uint8_t> ref = {
            144, 77,  101, 204, 139, 26,  245, 195, 154, 153, 25,  62,  92,
            113, 139, 234, 154, 153, 25,  62,  248, 146, 186, 205, 154, 153,
            25,  62,  100, 192, 21,  214, 154, 153, 25,  62,  195, 101, 233,
            196, 0,   177, 4,   228, 172, 38,  108, 235, 154, 153, 25,  62,
            175, 231, 130, 89,  154, 153, 25,  62,  41,  206, 132, 217, 218,
            221, 255, 193, 154, 153, 25,  62,  128, 136, 25,  62,  185, 75,
            60,  91,  139, 24,  10,  225, 243, 71,  214, 227, 154, 153, 25,
            62,  186, 125, 10,  236, 27,  8,   73,  233};

    geometry::Image image;

    // test image dimensions
    int width = 5;
    int height = 5;
    int num_of_channels = 1;
    int bytes_per_channel = 4;

    image.Prepare(width, height, num_of_channels, bytes_per_channel);

    Rand(image.data_, 0, 255, 0);

    auto output = image.CreateFloatImage();

    output->LinearTransform(2.3, 0.15);

    EXPECT_FALSE(output->IsEmpty());
    EXPECT_EQ(width, output->width_);
    EXPECT_EQ(height, output->height_);
    EXPECT_EQ(num_of_channels, output->num_of_channels_);
    EXPECT_EQ(bytes_per_channel, output->bytes_per_channel_);
    ExpectEQ(ref, output->data_);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, ClipIntensity) {
    // reference data used to validate the filtering of an image
    vector<uint8_t> ref = {
            195, 245, 168, 62,  195, 245, 168, 62,  195, 245, 168, 62,  195,
            245, 168, 62,  195, 245, 168, 62,  195, 245, 168, 62,  195, 245,
            168, 62,  195, 245, 168, 62,  195, 245, 168, 62,  195, 245, 168,
            62,  195, 245, 168, 62,  195, 245, 168, 62,  195, 245, 168, 62,
            143, 194, 53,  63,  195, 245, 168, 62,  195, 245, 168, 62,  195,
            245, 168, 62,  195, 245, 168, 62,  195, 245, 168, 62,  143, 194,
            53,  63,  195, 245, 168, 62,  195, 245, 168, 62,  195, 245, 168,
            62,  195, 245, 168, 62,  195, 245, 168, 62};

    geometry::Image image;

    // test image dimensions
    int width = 5;
    int height = 5;
    int num_of_channels = 1;
    int bytes_per_channel = 4;

    image.Prepare(width, height, num_of_channels, bytes_per_channel);

    Rand(image.data_, 0, 255, 0);

    auto output = image.CreateFloatImage();

    output->ClipIntensity(0.33, 0.71);

    EXPECT_FALSE(output->IsEmpty());
    EXPECT_EQ(width, output->width_);
    EXPECT_EQ(height, output->height_);
    EXPECT_EQ(num_of_channels, output->num_of_channels_);
    EXPECT_EQ(bytes_per_channel, output->bytes_per_channel_);
    ExpectEQ(ref, output->data_);
}

// ----------------------------------------------------------------------------
// Tests one of the following configurations
// ----------------------------------------------------------------------------
template <typename T>
void TEST_CreateImageFromFloatImage() {
    geometry::Image image;

    // test image dimensions
    int width = 10;
    int height = 10;
    int num_of_channels = 1;
    int bytes_per_channel = sizeof(T);

    image.Prepare(width, height, num_of_channels, bytes_per_channel);

    Rand(image.data_, 0, 255, 0);

    auto float_image = image.CreateFloatImage();

    auto output = float_image->CreateImageFromFloatImage<T>();

    EXPECT_FALSE(output->IsEmpty());
    EXPECT_EQ(width, output->width_);
    EXPECT_EQ(height, output->height_);
    EXPECT_EQ(num_of_channels, output->num_of_channels_);
    EXPECT_EQ(bytes_per_channel, output->bytes_per_channel_);
    ExpectEQ(image.data_, output->data_);
}

template void TEST_CreateImageFromFloatImage<uint8_t>();
template void TEST_CreateImageFromFloatImage<uint16_t>();

// ----------------------------------------------------------------------------
// Tests the case output image bytes_per_channel = 1.
// ----------------------------------------------------------------------------
TEST(Image, CreateImageFromFloatImage_8bit) {
    TEST_CreateImageFromFloatImage<uint8_t>();
}

// ----------------------------------------------------------------------------
// Tests the case output image bytes_per_channel = 2.
// ----------------------------------------------------------------------------
TEST(Image, CreateImageFromFloatImage_16bit) {
    TEST_CreateImageFromFloatImage<uint16_t>();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, FilterPyramid) {
    // reference data used to validate the filtering of an image
    vector<vector<uint8_t>> ref = {
            {110, 56,  130, 211, 17,  56,  2,   212, 198, 225, 181, 232,
             173, 226, 53,  233, 84,  159, 65,  233, 105, 3,   154, 233,
             112, 151, 223, 86,  113, 151, 95,  87,  93,  130, 242, 231,
             147, 137, 114, 232, 47,  173, 107, 233, 105, 3,   26,  234,
             224, 16,  192, 88,  171, 250, 111, 222, 215, 213, 65,  225,
             189, 203, 23,  226, 233, 196, 171, 233, 217, 210, 128, 234,
             47,  127, 9,   233, 201, 240, 161, 108, 7,   103, 35,  109,
             31,  224, 163, 108, 22,  49,  241, 233, 69,  228, 180, 234,
             36,  127, 137, 233, 201, 240, 33,  109, 9,   103, 163, 109,
             36,  224, 35,  109, 54,  50,  114, 233, 29,  165, 53,  234,
             237, 126, 9,   233, 202, 240, 161, 108, 9,   103, 35,  109,
             37,  224, 163, 108, 141, 106, 43,  229, 234, 143, 0,   230},
            {57,  48,  241, 106, 168, 116, 5,   107, 106, 200, 26,  106,
             115, 252, 23,  108, 93,  29,  48,  108, 107, 19,  140, 107,
             48,  19,  152, 108, 201, 182, 177, 108, 145, 200, 20,  108}};

    geometry::Image image;

    // test image dimensions
    int width = 6;
    int height = 6;
    int num_of_channels = 1;
    int bytes_per_channel = 4;
    int num_of_levels = 2;

    image.Prepare(width, height, num_of_channels, bytes_per_channel);

    Rand(image.data_, 0, 255, 0);

    auto float_image = image.CreateFloatImage();

    auto pyramid = float_image->CreatePyramid(num_of_levels);

    auto output_pyramid =
            geometry::Image::FilterPyramid(pyramid, FilterType::Gaussian3);

    EXPECT_EQ(pyramid.size(), output_pyramid.size());

    for (size_t p = 0; p < pyramid.size(); p++) {
        auto input = pyramid[p];
        auto output = output_pyramid[p];

        EXPECT_FALSE(output->IsEmpty());
        EXPECT_EQ(input->width_, output->width_);
        EXPECT_EQ(input->height_, output->height_);
        EXPECT_EQ(input->num_of_channels_, output->num_of_channels_);
        EXPECT_EQ(input->bytes_per_channel_, output->bytes_per_channel_);
        ExpectEQ(ref[p], output->data_);
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Image, CreatePyramid) {
    // reference data used to validate the filtering of an image
    vector<vector<uint8_t>> ref = {
            {214, 100, 199, 203, 232, 50,  85,  195, 70,  141, 121, 160,
             93,  130, 242, 233, 162, 182, 36,  154, 4,   61,  34,  205,
             39,  102, 33,  27,  254, 55,  130, 213, 156, 75,  162, 133,
             125, 248, 74,  196, 134, 196, 102, 227, 72,  89,  205, 234,
             17,  242, 134, 21,  49,  169, 227, 88,  16,  5,   116, 16,
             60,  247, 230, 216, 67,  137, 95,  193, 130, 170, 135, 10,
             111, 237, 237, 183, 72,  188, 163, 90,  175, 42,  112, 224,
             211, 84,  58,  227, 89,  175, 243, 150, 167, 218, 112, 235,
             101, 207, 174, 232, 123, 55,  242, 234, 37,  224, 163, 110,
             157, 71,  200, 78,  113, 57,  47,  70,  141, 106, 43,  231,
             26,  32,  126, 193, 251, 238, 174, 97,  191, 94,  75,  59,
             149, 62,  38,  186, 31,  202, 41,  189, 19,  242, 13,  132},
            {236, 42,  166, 86,  32, 227, 181, 232, 31, 44,  169, 233,
             203, 221, 160, 107, 20, 87,  117, 108, 78, 122, 78,  234,
             177, 76,  113, 108, 85, 1,   56,  109, 21, 221, 114, 233}};

    geometry::Image image;

    // test image dimensions
    int width = 6;
    int height = 6;
    int num_of_channels = 1;
    int bytes_per_channel = 4;
    int num_of_levels = 2;

    image.Prepare(width, height, num_of_channels, bytes_per_channel);

    Rand(image.data_, 0, 255, 0);

    auto float_image = image.CreateFloatImage();

    auto pyramid = float_image->CreatePyramid(num_of_levels);

    int expected_width = width;
    int expected_height = width;
    for (size_t p = 0; p < pyramid.size(); p++) {
        auto output = pyramid[p];

        EXPECT_FALSE(output->IsEmpty());
        EXPECT_EQ(expected_width, output->width_);
        EXPECT_EQ(expected_height, output->height_);
        EXPECT_EQ(num_of_channels, output->num_of_channels_);
        EXPECT_EQ(bytes_per_channel, output->bytes_per_channel_);
        ExpectEQ(ref[p], output->data_);

        expected_width /= 2;
        expected_height /= 2;
    }
}
