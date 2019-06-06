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

#include <vector>

#include "Open3D/Geometry/Image.h"
#include "Open3D/Geometry/RGBDImage.h"
#include "TestUtility/UnitTest.h"

using namespace open3d;
using namespace std;
using namespace unit_test;
using namespace unit_test;

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(RGBDImage, Constructor) {
    geometry::Image image;
    geometry::Image color;

    const int size = 5;

    // test image dimensions
    const int image_width = size;
    const int image_height = size;
    const int image_num_of_channels = 1;
    const int image_bytes_per_channel = 1;

    const int color_width = size;
    const int color_height = size;
    const int color_num_of_channels = 3;
    const int color_bytes_per_channel = 1;

    color.Prepare(color_width, color_height, color_num_of_channels,
                  color_bytes_per_channel);

    image.Prepare(image_width, image_height, image_num_of_channels,
                  image_bytes_per_channel);

    Rand(image.data_, 100, 150, 0);
    Rand(color.data_, 130, 200, 0);

    auto depth = image.ConvertDepthToFloatImage();

    geometry::RGBDImage rgbd_image(color, *depth);

    ExpectEQ(color.data_, rgbd_image.color_.data_);
    ExpectEQ(depth->data_, rgbd_image.depth_.data_);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(RGBDImage, DISABLED_MemberData) { unit_test::NotImplemented(); }

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(RGBDImage, CreateFromColorAndDepth) {
    vector<uint8_t> ref_color = {
            216, 2,   42,  63,  21,  162, 57,  63,  62,  210, 42,  63,  216,
            72,  38,  63,  116, 49,  38,  63,  55,  245, 52,  63,  150, 19,
            30,  63,  123, 6,   19,  63,  193, 10,  23,  63,  83,  253, 46,
            63,  141, 0,   52,  63,  9,   144, 38,  63,  193, 19,  55,  63,
            19,  179, 45,  63,  56,  160, 49,  63,  26,  10,  50,  63,  121,
            185, 46,  63,  168, 239, 16,  63,  183, 184, 35,  63,  63,  137,
            21,  63,  135, 1,   54,  63,  220, 15,  35,  63,  177, 246, 44,
            63,  207, 89,  38,  63,  56,  66,  57,  63};

    vector<uint8_t> ref_depth = {
            208, 254, 91,  58,  103, 154, 205, 57,  59,  147, 76,  58,  236,
            175, 80,  58,  232, 127, 110, 58,  103, 154, 77,  57,  62,  195,
            174, 57,  139, 118, 72,  58,  22,  236, 143, 57,  66,  243, 16,
            58,  161, 199, 248, 57,  134, 123, 36,  58,  255, 53,  191, 57,
            93,  164, 5,   58,  161, 199, 120, 58,  20,  135, 111, 58,  222,
            137, 38,  58,  79,  25,  59,  58,  198, 8,   20,  57,  126, 80,
            30,  58,  5,   150, 131, 55,  249, 213, 122, 57,  101, 207, 11,
            57,  68,  190, 82,  58,  214, 94,  32,  57};

    geometry::Image depth;
    geometry::Image color;

    const int size = 5;

    // test image dimensions
    const int depth_width = size;
    const int depth_height = size;
    const int depth_num_of_channels = 1;
    const int depth_bytes_per_channel = 4;

    const int color_width = size;
    const int color_height = size;
    const int color_num_of_channels = 3;
    const int color_bytes_per_channel = 1;

    color.Prepare(color_width, color_height, color_num_of_channels,
                  color_bytes_per_channel);

    depth.Prepare(depth_width, depth_height, depth_num_of_channels,
                  depth_bytes_per_channel);

    float* const float_data = Cast<float>(&depth.data_[0]);
    Rand(float_data, depth_width * depth_height, 0.0, 1.0, 0);
    Rand(color.data_, 130, 200, 0);

    auto rgbd_image =
            geometry::RGBDImage::CreateFromColorAndDepth(color, depth);

    ExpectEQ(ref_color, rgbd_image->color_.data_);
    ExpectEQ(ref_depth, rgbd_image->depth_.data_);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(RGBDImage, CreateFromRedwoodFormat) {
    vector<uint8_t> ref_color = {
            216, 2,   42,  63,  21,  162, 57,  63,  62,  210, 42,  63,  216,
            72,  38,  63,  116, 49,  38,  63,  55,  245, 52,  63,  150, 19,
            30,  63,  123, 6,   19,  63,  193, 10,  23,  63,  83,  253, 46,
            63,  141, 0,   52,  63,  9,   144, 38,  63,  193, 19,  55,  63,
            19,  179, 45,  63,  56,  160, 49,  63,  26,  10,  50,  63,  121,
            185, 46,  63,  168, 239, 16,  63,  183, 184, 35,  63,  63,  137,
            21,  63,  135, 1,   54,  63,  220, 15,  35,  63,  177, 246, 44,
            63,  207, 89,  38,  63,  56,  66,  57,  63};

    vector<uint8_t> ref_depth = {
            208, 254, 91,  58,  103, 154, 205, 57,  59,  147, 76,  58,  236,
            175, 80,  58,  232, 127, 110, 58,  103, 154, 77,  57,  62,  195,
            174, 57,  139, 118, 72,  58,  22,  236, 143, 57,  66,  243, 16,
            58,  161, 199, 248, 57,  134, 123, 36,  58,  255, 53,  191, 57,
            93,  164, 5,   58,  161, 199, 120, 58,  20,  135, 111, 58,  222,
            137, 38,  58,  79,  25,  59,  58,  198, 8,   20,  57,  126, 80,
            30,  58,  5,   150, 131, 55,  249, 213, 122, 57,  101, 207, 11,
            57,  68,  190, 82,  58,  214, 94,  32,  57};

    geometry::Image depth;
    geometry::Image color;

    const int size = 5;

    // test image dimensions
    const int depth_width = size;
    const int depth_height = size;
    const int depth_num_of_channels = 1;
    const int depth_bytes_per_channel = 4;

    const int color_width = size;
    const int color_height = size;
    const int color_num_of_channels = 3;
    const int color_bytes_per_channel = 1;

    color.Prepare(color_width, color_height, color_num_of_channels,
                  color_bytes_per_channel);

    depth.Prepare(depth_width, depth_height, depth_num_of_channels,
                  depth_bytes_per_channel);

    float* const float_data = Cast<float>(&depth.data_[0]);
    Rand(float_data, depth_width * depth_height, 0.0, 1.0, 0);
    Rand(color.data_, 130, 200, 0);

    auto rgbd_image =
            geometry::RGBDImage::CreateFromRedwoodFormat(color, depth);

    ExpectEQ(ref_color, rgbd_image->color_.data_);
    ExpectEQ(ref_depth, rgbd_image->depth_.data_);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(RGBDImage, CreateFromTUMFormat) {
    vector<uint8_t> ref_color = {
            216, 2,   42,  63,  21,  162, 57,  63,  62,  210, 42,  63,  216,
            72,  38,  63,  116, 49,  38,  63,  55,  245, 52,  63,  150, 19,
            30,  63,  123, 6,   19,  63,  193, 10,  23,  63,  83,  253, 46,
            63,  141, 0,   52,  63,  9,   144, 38,  63,  193, 19,  55,  63,
            19,  179, 45,  63,  56,  160, 49,  63,  26,  10,  50,  63,  121,
            185, 46,  63,  168, 239, 16,  63,  183, 184, 35,  63,  63,  137,
            21,  63,  135, 1,   54,  63,  220, 15,  35,  63,  177, 246, 44,
            63,  207, 89,  38,  63,  56,  66,  57,  63};

    vector<uint8_t> ref_depth = {
            13,  255, 47,  57,  134, 123, 164, 56,  252, 168, 35,  57,  35,
            243, 38,  57,  186, 204, 62,  57,  134, 123, 36,  56,  101, 207,
            139, 56,  214, 94,  32,  57,  137, 70,  102, 56,  156, 235, 231,
            56,  26,  6,   199, 56,  5,   150, 3,   57,  255, 247, 152, 56,
            200, 211, 213, 56,  26,  6,   71,  57,  68,  159, 63,  57,  24,
            59,  5,   57,  217, 173, 21,  57,  214, 218, 236, 55,  150, 77,
            253, 56,  162, 137, 82,  54,  45,  171, 72,  56,  60,  178, 223,
            55,  54,  152, 40,  57,  222, 75,  0,   56};

    geometry::Image depth;
    geometry::Image color;

    const int size = 5;

    // test image dimensions
    const int depth_width = size;
    const int depth_height = size;
    const int depth_num_of_channels = 1;
    const int depth_bytes_per_channel = 4;

    const int color_width = size;
    const int color_height = size;
    const int color_num_of_channels = 3;
    const int color_bytes_per_channel = 1;

    color.Prepare(color_width, color_height, color_num_of_channels,
                  color_bytes_per_channel);

    depth.Prepare(depth_width, depth_height, depth_num_of_channels,
                  depth_bytes_per_channel);

    float* const float_data = Cast<float>(&depth.data_[0]);
    Rand(float_data, depth_width * depth_height, 0.0, 1.0, 0);
    Rand(color.data_, 130, 200, 0);

    auto rgbd_image = geometry::RGBDImage::CreateFromTUMFormat(color, depth);

    ExpectEQ(ref_color, rgbd_image->color_.data_);
    ExpectEQ(ref_depth, rgbd_image->depth_.data_);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(RGBDImage, CreateFromSUNFormat) {
    vector<uint8_t> ref_color = {
            216, 2,   42,  63,  21,  162, 57,  63,  62,  210, 42,  63,  216,
            72,  38,  63,  116, 49,  38,  63,  55,  245, 52,  63,  150, 19,
            30,  63,  123, 6,   19,  63,  193, 10,  23,  63,  83,  253, 46,
            63,  141, 0,   52,  63,  9,   144, 38,  63,  193, 19,  55,  63,
            19,  179, 45,  63,  56,  160, 49,  63,  26,  10,  50,  63,  121,
            185, 46,  63,  168, 239, 16,  63,  183, 184, 35,  63,  63,  137,
            21,  63,  135, 1,   54,  63,  220, 15,  35,  63,  177, 246, 44,
            63,  207, 89,  38,  63,  56,  66,  57,  63};

    vector<uint8_t> ref_depth = {
            145, 158, 240, 194, 183, 111, 222, 2,   251, 170, 237, 226, 0,
            0,   0,   0,   181, 238, 242, 2,   105, 13,  206, 2,   0,   0,
            0,   0,   0,   0,   0,   0,   237, 185, 214, 130, 32,  61,  231,
            162, 0,   0,   0,   0,   41,  174, 233, 2,   253, 240, 190, 57,
            93,  164, 5,   58,  161, 199, 120, 58,  20,  135, 111, 58,  222,
            137, 38,  58,  79,  25,  59,  58,  198, 8,   20,  57,  126, 80,
            30,  58,  5,   150, 131, 55,  249, 213, 122, 57,  101, 207, 11,
            57,  68,  190, 82,  58,  214, 94,  32,  57};

    geometry::Image depth;
    geometry::Image color;

    const int size = 5;

    // test image dimensions
    const int depth_width = size;
    const int depth_height = size;
    const int depth_num_of_channels = 1;
    const int depth_bytes_per_channel = 4;

    const int color_width = size;
    const int color_height = size;
    const int color_num_of_channels = 3;
    const int color_bytes_per_channel = 1;

    color.Prepare(color_width, color_height, color_num_of_channels,
                  color_bytes_per_channel);

    depth.Prepare(depth_width, depth_height, depth_num_of_channels,
                  depth_bytes_per_channel);

    float* const float_data = Cast<float>(&depth.data_[0]);
    Rand(float_data, depth_width * depth_height, 0.0, 1.0, 0);
    Rand(color.data_, 130, 200, 0);

    auto rgbd_image = geometry::RGBDImage::CreateFromSUNFormat(color, depth);

    ExpectEQ(ref_color, rgbd_image->color_.data_);
    ExpectEQ(ref_depth, rgbd_image->depth_.data_);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(RGBDImage, CreateFromNYUFormat) {
    vector<uint8_t> ref_color = {
            216, 2,   42,  63,  21,  162, 57,  63,  62,  210, 42,  63,  216,
            72,  38,  63,  116, 49,  38,  63,  55,  245, 52,  63,  150, 19,
            30,  63,  123, 6,   19,  63,  193, 10,  23,  63,  83,  253, 46,
            63,  141, 0,   52,  63,  9,   144, 38,  63,  193, 19,  55,  63,
            19,  179, 45,  63,  56,  160, 49,  63,  26,  10,  50,  63,  121,
            185, 46,  63,  168, 239, 16,  63,  183, 184, 35,  63,  63,  137,
            21,  63,  135, 1,   54,  63,  220, 15,  35,  63,  177, 246, 44,
            63,  207, 89,  38,  63,  56,  66,  57,  63};

    vector<uint8_t> ref_depth = {
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   201, 118, 190, 57,
            93,  164, 5,   58,  161, 199, 120, 58,  20,  135, 111, 58,  222,
            137, 38,  58,  79,  25,  59,  58,  198, 8,   20,  57,  126, 80,
            30,  58,  5,   150, 131, 55,  249, 213, 122, 57,  101, 207, 11,
            57,  68,  190, 82,  58,  214, 94,  32,  57};

    geometry::Image depth;
    geometry::Image color;

    const int size = 5;

    // test image dimensions
    const int depth_width = size;
    const int depth_height = size;
    const int depth_num_of_channels = 1;
    const int depth_bytes_per_channel = 4;

    const int color_width = size;
    const int color_height = size;
    const int color_num_of_channels = 3;
    const int color_bytes_per_channel = 1;

    color.Prepare(color_width, color_height, color_num_of_channels,
                  color_bytes_per_channel);

    depth.Prepare(depth_width, depth_height, depth_num_of_channels,
                  depth_bytes_per_channel);

    float* const float_data = Cast<float>(&depth.data_[0]);
    Rand(float_data, depth_width * depth_height, 0.0, 1.0, 0);
    Rand(color.data_, 130, 200, 0);

    auto rgbd_image = geometry::RGBDImage::CreateFromNYUFormat(color, depth);

    ExpectEQ(ref_color, rgbd_image->color_.data_);
    ExpectEQ(ref_depth, rgbd_image->depth_.data_);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(RGBDImage, FilterPyramid) {
    vector<vector<uint8_t>> ref_color = {
            {49,  63,  46,  63,  234, 198, 45,  63,  152, 189, 39,  63,  151,
             141, 36,  63,  165, 233, 38,  63,  44,  66,  47,  63,  54,  137,
             40,  63,  10,  229, 34,  63,  34,  30,  36,  63,  102, 55,  42,
             63,  230, 110, 48,  63,  133, 79,  41,  63,  33,  69,  37,  63,
             126, 99,  38,  63,  234, 215, 40,  63,  91,  21,  49,  63,  92,
             34,  42,  63,  75,  230, 36,  63,  181, 45,  37,  63,  210, 213,
             37,  63,  82,  65,  49,  63,  72,  187, 41,  63,  106, 229, 37,
             63,  255, 106, 40,  63,  94,  171, 45,  63},
            {159, 3, 43, 63, 135, 253, 38, 63, 128, 50, 43, 63, 2, 65, 39, 63}};

    vector<vector<uint8_t>> ref_depth = {
            {38,  31,  30,  58,  91,  210, 16,  58,  248, 34,  42,  58,  238,
             234, 63,  58,  236, 245, 76,  58,  110, 243, 222, 57,  98,  12,
             254, 57,  47,  168, 17,  58,  162, 44,  25,  58,  168, 28,  48,
             58,  39,  160, 9,   58,  4,   196, 14,  58,  243, 74,  4,   58,
             173, 65,  5,   58,  160, 171, 45,  58,  11,  239, 20,  58,  154,
             141, 11,  58,  214, 133, 248, 57,  250, 97,  243, 57,  128, 174,
             11,  58,  56,  200, 138, 57,  214, 164, 156, 57,  33,  250, 200,
             57,  206, 160, 238, 57,  123, 32,  188, 57},
            {196, 181, 8, 58, 174, 43, 22, 58, 190, 83, 23, 58, 152, 174, 7,
             58}};

    geometry::Image depth;
    geometry::Image color;

    const int size = 5;

    // test image dimensions
    const int depth_width = size;
    const int depth_height = size;
    const int depth_num_of_channels = 1;
    const int depth_bytes_per_channel = 4;

    const int color_width = size;
    const int color_height = size;
    const int color_num_of_channels = 3;
    const int color_bytes_per_channel = 1;

    color.Prepare(color_width, color_height, color_num_of_channels,
                  color_bytes_per_channel);

    depth.Prepare(depth_width, depth_height, depth_num_of_channels,
                  depth_bytes_per_channel);

    float* const float_data = Cast<float>(&depth.data_[0]);
    Rand(float_data, depth_width * depth_height, 0.0, 1.0, 0);
    Rand(color.data_, 130, 200, 0);

    size_t num_of_levels = 2;
    auto rgbd_image =
            geometry::RGBDImage::CreateFromColorAndDepth(color, depth);
    auto pyramid = rgbd_image->CreatePyramid(num_of_levels);
    auto filtered = geometry::RGBDImage::FilterPyramid(
            pyramid, geometry::Image::FilterType::Gaussian3);

    for (size_t j = 0; j < num_of_levels; j++) {
        ExpectEQ(ref_color[j], filtered[j]->color_.data_);
        ExpectEQ(ref_depth[j], filtered[j]->depth_.data_);
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(RGBDImage, CreatePyramid) {
    vector<vector<uint8_t>> ref_color = {
            {216, 2,   42,  63,  21,  162, 57,  63,  62,  210, 42,  63,  216,
             72,  38,  63,  116, 49,  38,  63,  55,  245, 52,  63,  150, 19,
             30,  63,  123, 6,   19,  63,  193, 10,  23,  63,  83,  253, 46,
             63,  141, 0,   52,  63,  9,   144, 38,  63,  193, 19,  55,  63,
             19,  179, 45,  63,  56,  160, 49,  63,  26,  10,  50,  63,  121,
             185, 46,  63,  168, 239, 16,  63,  183, 184, 35,  63,  63,  137,
             21,  63,  135, 1,   54,  63,  220, 15,  35,  63,  177, 246, 44,
             63,  207, 89,  38,  63,  56,  66,  57,  63},
            {96, 244, 44, 63, 151, 211, 36, 63, 137, 61, 45, 63, 40, 111, 37,
             63}};

    vector<vector<uint8_t>> ref_depth = {
            {208, 254, 91,  58,  103, 154, 205, 57,  59,  147, 76,  58,  236,
             175, 80,  58,  232, 127, 110, 58,  103, 154, 77,  57,  62,  195,
             174, 57,  139, 118, 72,  58,  22,  236, 143, 57,  66,  243, 16,
             58,  161, 199, 248, 57,  134, 123, 36,  58,  255, 53,  191, 57,
             93,  164, 5,   58,  161, 199, 120, 58,  20,  135, 111, 58,  222,
             137, 38,  58,  79,  25,  59,  58,  198, 8,   20,  57,  126, 80,
             30,  58,  5,   150, 131, 55,  249, 213, 122, 57,  101, 207, 11,
             57,  68,  190, 82,  58,  214, 94,  32,  57},
            {30, 202, 230, 57, 240, 107, 43, 58, 18, 188, 45, 58, 111, 173, 226,
             57}};

    geometry::Image depth;
    geometry::Image color;

    const int size = 5;

    // test image dimensions
    const int depth_width = size;
    const int depth_height = size;
    const int depth_num_of_channels = 1;
    const int depth_bytes_per_channel = 4;

    const int color_width = size;
    const int color_height = size;
    const int color_num_of_channels = 3;
    const int color_bytes_per_channel = 1;

    color.Prepare(color_width, color_height, color_num_of_channels,
                  color_bytes_per_channel);

    depth.Prepare(depth_width, depth_height, depth_num_of_channels,
                  depth_bytes_per_channel);

    float* const float_data = Cast<float>(&depth.data_[0]);
    Rand(float_data, depth_width * depth_height, 0.0, 1.0, 0);
    Rand(color.data_, 130, 200, 0);

    size_t num_of_levels = 2;
    auto rgbd_image =
            geometry::RGBDImage::CreateFromColorAndDepth(color, depth);
    auto pyramid = rgbd_image->CreatePyramid(num_of_levels);

    for (size_t j = 0; j < num_of_levels; j++) {
        EXPECT_EQ(ref_color[j], pyramid[j]->color_.data_);
        EXPECT_EQ(ref_depth[j], pyramid[j]->depth_.data_);
    }
}
