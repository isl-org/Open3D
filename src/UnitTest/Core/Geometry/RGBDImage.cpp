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

#include <Core/Geometry/Image.h>
#include <Core/Geometry/RGBDImage.h>

#include <vector>

using namespace std;

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(RGBDImage, Constructor)
{
    open3d::Image image;
    open3d::Image color;

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

    color.PrepareImage(color_width,
                       color_height,
                       color_num_of_channels,
                       color_bytes_per_channel);

    image.PrepareImage(image_width,
                       image_height,
                       image_num_of_channels,
                       image_bytes_per_channel);

    UnitTest::Rand<uint8_t>(image.data_, 100, 150, 0);
    UnitTest::Rand<uint8_t>(color.data_, 130, 200, 0);

    auto depth = open3d::ConvertDepthToFloatImage(image);

    open3d::RGBDImage rgbdImage(color, *depth);

    for (size_t i = 0; i < color.data_.size(); i++)
        EXPECT_EQ(color.data_[i], rgbdImage.color_.data_[i]);

    for (size_t i = 0; i < depth->data_.size(); i++)
        EXPECT_EQ(depth->data_[i], rgbdImage.depth_.data_[i]);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(RGBDImage, DISABLED_MemberData)
{
    UnitTest::NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(RGBDImage, CreateRGBDImageFromColorAndDepth)
{
    vector<uint8_t> ref_color =
    { 216,    2,   42,   63,   21,  162,   57,   63,   62,  210,\
       42,   63,  216,   72,   38,   63,  116,   49,   38,   63,\
       55,  245,   52,   63,  150,   19,   30,   63,  123,    6,\
       19,   63,  193,   10,   23,   63,   83,  253,   46,   63,\
      141,    0,   52,   63,    9,  144,   38,   63,  193,   19,\
       55,   63,   19,  179,   45,   63,   56,  160,   49,   63,\
       26,   10,   50,   63,  121,  185,   46,   63,  168,  239,\
       16,   63,  183,  184,   35,   63,   63,  137,   21,   63,\
      135,    1,   54,   63,  220,   15,   35,   63,  177,  246,\
       44,   63,  207,   89,   38,   63,   56,   66,   57,   63 };

    vector<uint8_t> ref_depth =
    { 172,  219,   92,   54,  209,  104,  206,   53,  157,   96,\
       77,   54,  110,  129,   81,   54,   89,  111,  111,   54,\
      209,  104,   78,   53,  178,  114,  175,   53,  204,   63,\
       73,   54,  146,  124,  144,   53,  199,  132,   17,   54,\
       99,  193,  249,   53,  168,   32,   37,   54,  246,  245,\
      191,   53,  136,   42,    6,   54,   99,  193,  121,   54,\
      141,  119,  112,   54,   16,   49,   39,   54,   37,  213,\
       59,   54,   99,  157,   20,   53,  110,  239,   30,   54,\
       32,   26,  132,   51,  204,  209,  123,   53,  194,   91,\
       12,   53,  214,  145,   83,   54,  214,  255,   32,   53 };

    open3d::Image depth;
    open3d::Image color;

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

    color.PrepareImage(color_width,
                       color_height,
                       color_num_of_channels,
                       color_bytes_per_channel);

    depth.PrepareImage(depth_width,
                       depth_height,
                       depth_num_of_channels,
                       depth_bytes_per_channel);

    float* const depthFloatData = reinterpret_cast<float*>(&depth.data_[0]);
    UnitTest::Rand<float>(depthFloatData, depth_width * depth_height, 0.0, 1.0, 0);
    UnitTest::Rand<uint8_t>(color.data_, 130, 200, 0);

    UnitTest::Print(depth.data_);

    auto rgbdImage = open3d::CreateRGBDImageFromColorAndDepth(color, depth);

    UnitTest::Print(rgbdImage->color_.data_);
    UnitTest::Print(rgbdImage->depth_.data_);

    EXPECT_EQ(ref_color.size(), rgbdImage->color_.data_.size());
    for (size_t i = 0; i < color.data_.size(); i++)
        EXPECT_EQ(ref_color[i], rgbdImage->color_.data_[i]);

    EXPECT_EQ(ref_depth.size(), rgbdImage->depth_.data_.size());
    for (size_t i = 0; i < depth.data_.size(); i++)
        EXPECT_EQ(ref_depth[i], rgbdImage->depth_.data_[i]);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(RGBDImage, CreateRGBDImageFromRedwoodFormat)
{
    vector<uint8_t> ref_color =
    { 216,    2,   42,   63,   21,  162,   57,   63,   62,  210,\
       42,   63,  216,   72,   38,   63,  116,   49,   38,   63,\
       55,  245,   52,   63,  150,   19,   30,   63,  123,    6,\
       19,   63,  193,   10,   23,   63,   83,  253,   46,   63,\
      141,    0,   52,   63,    9,  144,   38,   63,  193,   19,\
       55,   63,   19,  179,   45,   63,   56,  160,   49,   63,\
       26,   10,   50,   63,  121,  185,   46,   63,  168,  239,\
       16,   63,  183,  184,   35,   63,   63,  137,   21,   63,\
      135,    1,   54,   63,  220,   15,   35,   63,  177,  246,\
       44,   63,  207,   89,   38,   63,   56,   66,   57,   63 };

    vector<uint8_t> ref_depth =
    { 213,  109,   20,   53,   46,  138,  250,   52,  218,   82,\
       18,   53,  218,   82,   18,   53,  201,  163,   24,   53,\
      106,  124,  229,   52,   63,   57,  244,   52,   93,   69,\
       17,   53,   82,  232,  237,   52,  255,  176,    5,   53,\
       11,  123,    1,   53,  243,  230,    9,   53,   52,  111,\
      248,   52,    5,  150,    3,   53,  195,  190,   26,   53,\
      201,  163,   24,   53,  243,  230,    9,   53,  230,   28,\
       14,   53,  119,   70,  225,   52,  118,  217,    8,   53,\
      161,  137,  210,   52,   94,  178,  233,   52,  125,   43,\
      223,   52,   88,   96,   19,   53,  119,   70,  225,   52 };

    open3d::Image image;
    open3d::Image color;

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

    color.PrepareImage(color_width,
                       color_height,
                       color_num_of_channels,
                       color_bytes_per_channel);

    image.PrepareImage(image_width,
                       image_height,
                       image_num_of_channels,
                       image_bytes_per_channel);

    UnitTest::Rand<uint8_t>(image.data_, 100, 150, 0);
    UnitTest::Rand<uint8_t>(color.data_, 130, 200, 0);

    auto depth = open3d::ConvertDepthToFloatImage(image);

    auto rgbdImage = open3d::CreateRGBDImageFromRedwoodFormat(color, *depth);

    EXPECT_EQ(ref_color.size(), rgbdImage->color_.data_.size());
    for (size_t i = 0; i < color.data_.size(); i++)
        EXPECT_EQ(ref_color[i], rgbdImage->color_.data_[i]);

    EXPECT_EQ(ref_depth.size(), rgbdImage->depth_.data_.size());
    for (size_t i = 0; i < depth->data_.size(); i++)
        EXPECT_EQ(ref_depth[i], rgbdImage->depth_.data_[i]);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(RGBDImage, CreateRGBDImageFromTUMFormat)
{
    vector<uint8_t> ref_color =
    { 216,    2,   42,   63,   21,  162,   57,   63,   62,  210,\
       42,   63,  216,   72,   38,   63,  116,   49,   38,   63,\
       55,  245,   52,   63,  150,   19,   30,   63,  123,    6,\
       19,   63,  193,   10,   23,   63,   83,  253,   46,   63,\
      141,    0,   52,   63,    9,  144,   38,   63,  193,   19,\
       55,   63,   19,  179,   45,   63,   56,  160,   49,   63,\
       26,   10,   50,   63,  121,  185,   46,   63,  168,  239,\
       16,   63,  183,  184,   35,   63,   63,  137,   21,   63,\
      135,    1,   54,   63,  220,   15,   35,   63,  177,  246,\
       44,   63,  207,   89,   38,   63,   56,   66,   57,   63 };

    vector<uint8_t> ref_depth =
    { 136,  124,  237,   51,  139,  110,  200,   51,   42,   30,\
      234,   51,   42,   30,  234,   51,   65,   57,  244,   51,\
      188,  150,  183,   51,    0,   97,  195,   51,  251,  110,\
      232,   51,  117,   83,  190,   51,  254,  231,  213,   51,\
       68,   43,  207,   51,  184,  164,  220,   51,   93,  191,\
      198,   51,  162,  137,  210,   51,  158,  151,  247,   51,\
       65,   57,  244,   51,  184,  164,  220,   51,  113,   97,\
      227,   51,   95,   56,  180,   51,  137,  245,  218,   51,\
       26,  110,  168,   51,   24,  245,  186,   51,   49,  137,\
      178,   51,   90,  205,  235,   51,   95,   56,  180,   51 };

    open3d::Image image;
    open3d::Image color;

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

    color.PrepareImage(color_width,
                       color_height,
                       color_num_of_channels,
                       color_bytes_per_channel);

    image.PrepareImage(image_width,
                       image_height,
                       image_num_of_channels,
                       image_bytes_per_channel);

    UnitTest::Rand<uint8_t>(image.data_, 100, 150, 0);
    UnitTest::Rand<uint8_t>(color.data_, 130, 200, 0);

    auto depth = open3d::ConvertDepthToFloatImage(image);

    auto rgbdImage = open3d::CreateRGBDImageFromTUMFormat(color, *depth);

    EXPECT_EQ(ref_color.size(), rgbdImage->color_.data_.size());
    for (size_t i = 0; i < color.data_.size(); i++)
        EXPECT_EQ(ref_color[i], rgbdImage->color_.data_[i]);

    EXPECT_EQ(ref_depth.size(), rgbdImage->depth_.data_.size());
    for (size_t i = 0; i < depth->data_.size(); i++)
        EXPECT_EQ(ref_depth[i], rgbdImage->depth_.data_[i]);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(RGBDImage, CreateRGBDImageFromSUNFormat)
{
    vector<uint8_t> ref_color =
    { 216,    2,   42,   63,   21,  162,   57,   63,   62,  210,\
       42,   63,  216,   72,   38,   63,  116,   49,   38,   63,\
       55,  245,   52,   63,  150,   19,   30,   63,  123,    6,\
       19,   63,  193,   10,   23,   63,   83,  253,   46,   63,\
      141,    0,   52,   63,    9,  144,   38,   63,  193,   19,\
       55,   63,   19,  179,   45,   63,   56,  160,   49,   63,\
       26,   10,   50,   63,  121,  185,   46,   63,  168,  239,\
       16,   63,  183,  184,   35,   63,   63,  137,   21,   63,\
      135,    1,   54,   63,  220,   15,   35,   63,  177,  246,\
       44,   63,  207,   89,   38,   63,   56,   66,   57,   63 };

    vector<uint8_t> ref_depth =
    { 156,    8,   71,    2,    2,  198,   66,  130,  220,  223,\
       69,  194,  220,  223,   69,  194,  131,  235,   70,  162,\
      152,  134,   64,    2,  151,  153,   65,  194,    3,   98,\
       70,  162,   20,  148,   65,    2,    0,    0,    0,    0,\
      127,  211,   67,  130,   95,  210,   68,  194,  169,    3,\
      248,   52,    5,  150,    3,   53,  195,  190,   26,   53,\
      201,  163,   24,   53,  243,  230,    9,   53,  230,   28,\
       14,   53,  119,   70,  225,   52,  118,  217,    8,   53,\
      161,  137,  210,   52,   94,  178,  233,   52,  125,   43,\
      223,   52,   88,   96,   19,   53,  119,   70,  225,   52 };

    open3d::Image image;
    open3d::Image color;

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

    color.PrepareImage(color_width,
                       color_height,
                       color_num_of_channels,
                       color_bytes_per_channel);

    image.PrepareImage(image_width,
                       image_height,
                       image_num_of_channels,
                       image_bytes_per_channel);

    UnitTest::Rand<uint8_t>(image.data_, 100, 150, 0);
    UnitTest::Rand<uint8_t>(color.data_, 130, 200, 0);

    auto depth = open3d::ConvertDepthToFloatImage(image);

    auto rgbdImage = open3d::CreateRGBDImageFromSUNFormat(color, *depth);

    EXPECT_EQ(ref_color.size(), rgbdImage->color_.data_.size());
    for (size_t i = 0; i < color.data_.size(); i++)
        EXPECT_EQ(ref_color[i], rgbdImage->color_.data_[i]);

    EXPECT_EQ(ref_depth.size(), rgbdImage->depth_.data_.size());
    for (size_t i = 0; i < depth->data_.size(); i++)
        EXPECT_EQ(ref_depth[i], rgbdImage->depth_.data_[i]);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(RGBDImage, CreateRGBDImageFromNYUFormat)
{
    vector<uint8_t> ref_color =
    { 216,    2,   42,   63,   21,  162,   57,   63,   62,  210,\
       42,   63,  216,   72,   38,   63,  116,   49,   38,   63,\
       55,  245,   52,   63,  150,   19,   30,   63,  123,    6,\
       19,   63,  193,   10,   23,   63,   83,  253,   46,   63,\
      141,    0,   52,   63,    9,  144,   38,   63,  193,   19,\
       55,   63,   19,  179,   45,   63,   56,  160,   49,   63,\
       26,   10,   50,   63,  121,  185,   46,   63,  168,  239,\
       16,   63,  183,  184,   35,   63,   63,  137,   21,   63,\
      135,    1,   54,   63,  220,   15,   35,   63,  177,  246,\
       44,   63,  207,   89,   38,   63,   56,   66,   57,   63 };

    vector<uint8_t> ref_depth =
    { 156,    8,   71,    2,    2,  198,   66,  130,  220,  223,\
       69,  194,  220,  223,   69,  194,  131,  235,   70,  162,\
      152,  134,   64,    2,  151,  153,   65,  194,    3,   98,\
       70,  162,   20,  148,   65,    2,    0,    0,    0,    0,\
      127,  211,   67,  130,   95,  210,   68,  194,  169,    3,\
      248,   52,    5,  150,    3,   53,  195,  190,   26,   53,\
      201,  163,   24,   53,  243,  230,    9,   53,  230,   28,\
       14,   53,  119,   70,  225,   52,  118,  217,    8,   53,\
      161,  137,  210,   52,   94,  178,  233,   52,  125,   43,\
      223,   52,   88,   96,   19,   53,  119,   70,  225,   52 };

    open3d::Image image;
    open3d::Image color;

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

    color.PrepareImage(color_width,
                       color_height,
                       color_num_of_channels,
                       color_bytes_per_channel);

    image.PrepareImage(image_width,
                       image_height,
                       image_num_of_channels,
                       image_bytes_per_channel);

    UnitTest::Rand<uint8_t>(image.data_, 100, 150, 0);
    UnitTest::Rand<uint8_t>(color.data_, 130, 200, 0);

    auto depth = open3d::ConvertDepthToFloatImage(image);

    auto rgbdImage = open3d::CreateRGBDImageFromSUNFormat(color, *depth);

    EXPECT_EQ(ref_color.size(), rgbdImage->color_.data_.size());
    for (size_t i = 0; i < color.data_.size(); i++)
        EXPECT_EQ(ref_color[i], rgbdImage->color_.data_[i]);

    EXPECT_EQ(ref_depth.size(), rgbdImage->depth_.data_.size());
    for (size_t i = 0; i < depth->data_.size(); i++)
        EXPECT_EQ(ref_depth[i], rgbdImage->depth_.data_[i]);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(RGBDImage, DISABLED_FilterRGBDImagePyramid)
{
    UnitTest::NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(RGBDImage, DISABLED_CreateRGBDImagePyramid)
{
    UnitTest::NotImplemented();
}
