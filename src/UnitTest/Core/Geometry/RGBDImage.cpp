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

    unit_test::Rand(image.data_, 100, 150, 0);
    unit_test::Rand(color.data_, 130, 200, 0);

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
    unit_test::NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(RGBDImage, CreateRGBDImageFromColorAndDepth)
{
    vector<uint8_t> ref_color =
    {
          216,    2,   42,   63,   21,  162,   57,   63,   62,  210,
           42,   63,  216,   72,   38,   63,  116,   49,   38,   63,
           55,  245,   52,   63,  150,   19,   30,   63,  123,    6,
           19,   63,  193,   10,   23,   63,   83,  253,   46,   63,
          141,    0,   52,   63,    9,  144,   38,   63,  193,   19,
           55,   63,   19,  179,   45,   63,   56,  160,   49,   63,
           26,   10,   50,   63,  121,  185,   46,   63,  168,  239,
           16,   63,  183,  184,   35,   63,   63,  137,   21,   63,
          135,    1,   54,   63,  220,   15,   35,   63,  177,  246,
           44,   63,  207,   89,   38,   63,   56,   66,   57,   63
    };

    vector<uint8_t> ref_depth =
    {
          172,  219,   92,   54,  209,  104,  206,   53,  157,   96,
           77,   54,  110,  129,   81,   54,   89,  111,  111,   54,
          209,  104,   78,   53,  178,  114,  175,   53,  204,   63,
           73,   54,  146,  124,  144,   53,  199,  132,   17,   54,
           99,  193,  249,   53,  168,   32,   37,   54,  246,  245,
          191,   53,  136,   42,    6,   54,   99,  193,  121,   54,
          141,  119,  112,   54,   16,   49,   39,   54,   37,  213,
           59,   54,   99,  157,   20,   53,  110,  239,   30,   54,
           32,   26,  132,   51,  204,  209,  123,   53,  194,   91,
           12,   53,  214,  145,   83,   54,  214,  255,   32,   53
    };

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

    float* const float_data = reinterpret_cast<float*>(&depth.data_[0]);
    unit_test::Rand(float_data, depth_width * depth_height, 0.0, 1.0, 0);
    unit_test::Rand(color.data_, 130, 200, 0);

    auto rgbdImage = open3d::CreateRGBDImageFromColorAndDepth(color, depth);

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
    {
          216,    2,   42,   63,   21,  162,   57,   63,   62,  210,
           42,   63,  216,   72,   38,   63,  116,   49,   38,   63,
           55,  245,   52,   63,  150,   19,   30,   63,  123,    6,
           19,   63,  193,   10,   23,   63,   83,  253,   46,   63,
          141,    0,   52,   63,    9,  144,   38,   63,  193,   19,
           55,   63,   19,  179,   45,   63,   56,  160,   49,   63,
           26,   10,   50,   63,  121,  185,   46,   63,  168,  239,
           16,   63,  183,  184,   35,   63,   63,  137,   21,   63,
          135,    1,   54,   63,  220,   15,   35,   63,  177,  246,
           44,   63,  207,   89,   38,   63,   56,   66,   57,   63
    };

    vector<uint8_t> ref_depth =
    {
          172,  219,   92,   54,  209,  104,  206,   53,  157,   96,
           77,   54,  110,  129,   81,   54,   89,  111,  111,   54,
          209,  104,   78,   53,  178,  114,  175,   53,  204,   63,
           73,   54,  146,  124,  144,   53,  199,  132,   17,   54,
           99,  193,  249,   53,  168,   32,   37,   54,  246,  245,
          191,   53,  136,   42,    6,   54,   99,  193,  121,   54,
          141,  119,  112,   54,   16,   49,   39,   54,   37,  213,
           59,   54,   99,  157,   20,   53,  110,  239,   30,   54,
           32,   26,  132,   51,  204,  209,  123,   53,  194,   91,
           12,   53,  214,  145,   83,   54,  214,  255,   32,   53
    };

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

    float* const float_data = reinterpret_cast<float*>(&depth.data_[0]);
    unit_test::Rand(float_data, depth_width * depth_height, 0.0, 1.0, 0);
    unit_test::Rand(color.data_, 130, 200, 0);

    auto rgbdImage = open3d::CreateRGBDImageFromRedwoodFormat(color, depth);

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
TEST(RGBDImage, CreateRGBDImageFromTUMFormat)
{
    vector<uint8_t> ref_color =
    {
          216,    2,   42,   63,   21,  162,   57,   63,   62,  210,
           42,   63,  216,   72,   38,   63,  116,   49,   38,   63,
           55,  245,   52,   63,  150,   19,   30,   63,  123,    6,
           19,   63,  193,   10,   23,   63,   83,  253,   46,   63,
          141,    0,   52,   63,    9,  144,   38,   63,  193,   19,
           55,   63,   19,  179,   45,   63,   56,  160,   49,   63,
           26,   10,   50,   63,  121,  185,   46,   63,  168,  239,
           16,   63,  183,  184,   35,   63,   63,  137,   21,   63,
          135,    1,   54,   63,  220,   15,   35,   63,  177,  246,
           44,   63,  207,   89,   38,   63,   56,   66,   57,   63
    };

    vector<uint8_t> ref_depth =
    {
          189,  175,   48,   53,  167,   32,  165,   52,   74,   77,
           36,   53,  190,  154,   39,   53,   71,  140,   63,   53,
          167,   32,   37,   52,  194,   91,  140,   52,  214,  255,
           32,   53,  183,   45,  103,   52,  113,  212,  232,   52,
          233,  205,  199,   52,   32,   26,    4,   53,  145,  145,
          153,   52,  116,  170,  214,   52,  233,  205,   71,   53,
          164,   95,   64,   53,  218,  192,    5,   53,   29,   68,
           22,   53,  159,  200,  237,   51,  227,   75,  254,   52,
            0,   93,   83,   50,  163,  116,   73,   52,  207,  146,
          224,   51,  120,   65,   41,   53,  171,  204,    0,   52
    };

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

    float* const float_data = reinterpret_cast<float*>(&depth.data_[0]);
    unit_test::Rand(float_data, depth_width * depth_height, 0.0, 1.0, 0);
    unit_test::Rand(color.data_, 130, 200, 0);

    auto rgbdImage = open3d::CreateRGBDImageFromTUMFormat(color, depth);

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
TEST(RGBDImage, CreateRGBDImageFromSUNFormat)
{
    vector<uint8_t> ref_color =
    {
          216,    2,   42,   63,   21,  162,   57,   63,   62,  210,
           42,   63,  216,   72,   38,   63,  116,   49,   38,   63,
           55,  245,   52,   63,  150,   19,   30,   63,  123,    6,
           19,   63,  193,   10,   23,   63,   83,  253,   46,   63,
          141,    0,   52,   63,    9,  144,   38,   63,  193,   19,
           55,   63,   19,  179,   45,   63,   56,  160,   49,   63,
           26,   10,   50,   63,  121,  185,   46,   63,  168,  239,
           16,   63,  183,  184,   35,   63,   63,  137,   21,   63,
          135,    1,   54,   63,  220,   15,   35,   63,  177,  246,
           44,   63,  207,   89,   38,   63,   56,   66,   57,   63
    };

    vector<uint8_t> ref_depth =
    {
          164,  120,  112,  226,   13,  203,   94,   34,  204,  202,
          110,    2,  229,   78,  111,  130,  241,  110,  115,   34,
          191,  104,   78,   34,    0,    0,    0,    0,  177,   70,
          110,  130,  193,  236,   86,  162,  122,  171,  102,  194,
            0,    0,    0,    0,   86,    4,  106,   34,  152,  205,
          191,   53,  136,   42,    6,   54,   99,  193,  121,   54,
          141,  119,  112,   54,   16,   49,   39,   54,   37,  213,
           59,   54,   99,  157,   20,   53,  110,  239,   30,   54,
           32,   26,  132,   51,  204,  209,  123,   53,  194,   91,
           12,   53,  214,  145,   83,   54,  214,  255,   32,   53
    };

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

    float* const float_data = reinterpret_cast<float*>(&depth.data_[0]);
    unit_test::Rand(float_data, depth_width * depth_height, 0.0, 1.0, 0);
    unit_test::Rand(color.data_, 130, 200, 0);

    auto rgbdImage = open3d::CreateRGBDImageFromSUNFormat(color, depth);

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
TEST(RGBDImage, CreateRGBDImageFromNYUFormat)
{
    vector<uint8_t> ref_color =
    {
          216,    2,   42,   63,   21,  162,   57,   63,   62,  210,
           42,   63,  216,   72,   38,   63,  116,   49,   38,   63,
           55,  245,   52,   63,  150,   19,   30,   63,  123,    6,
           19,   63,  193,   10,   23,   63,   83,  253,   46,   63,
          141,    0,   52,   63,    9,  144,   38,   63,  193,   19,
           55,   63,   19,  179,   45,   63,   56,  160,   49,   63,
           26,   10,   50,   63,  121,  185,   46,   63,  168,  239,
           16,   63,  183,  184,   35,   63,   63,  137,   21,   63,
          135,    1,   54,   63,  220,   15,   35,   63,  177,  246,
           44,   63,  207,   89,   38,   63,   56,   66,   57,   63
    };

    vector<uint8_t> ref_depth =
    {
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    1,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,  238,  124,
          191,   53,  136,   42,    6,   54,   99,  193,  121,   54,
          141,  119,  112,   54,   16,   49,   39,   54,   37,  213,
           59,   54,   99,  157,   20,   53,  110,  239,   30,   54,
           32,   26,  132,   51,  204,  209,  123,   53,  194,   91,
           12,   53,  214,  145,   83,   54,  214,  255,   32,   53
    };

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

    float* const float_data = reinterpret_cast<float*>(&depth.data_[0]);
    unit_test::Rand(float_data, depth_width * depth_height, 0.0, 1.0, 0);
    unit_test::Rand(color.data_, 130, 200, 0);

    auto rgbdImage = open3d::CreateRGBDImageFromNYUFormat(color, depth);

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
TEST(RGBDImage, FilterRGBDImagePyramid)
{
    vector<vector<uint8_t>> ref_color =
    {
        {
               49,   63,   46,   63,  234,  198,   45,   63,  152,  189,
               39,   63,  151,  141,   36,   63,  165,  233,   38,   63,
               44,   66,   47,   63,   54,  137,   40,   63,   10,  229,
               34,   63,   34,   30,   36,   63,  102,   55,   42,   63,
              230,  110,   48,   63,  133,   79,   41,   63,   33,   69,
               37,   63,  126,   99,   38,   63,  234,  215,   40,   63,
               91,   21,   49,   63,   92,   34,   42,   63,   75,  230,
               36,   63,  181,   45,   37,   63,  210,  213,   37,   63,
               82,   65,   49,   63,   72,  187,   41,   63,  106,  229,
               37,   63,  255,  106,   40,   63,   94,  171,   45,   63
        },
        {
              159,    3,   43,   63,  135,  253,   38,   63,  128,   50,
               43,   63,    2,   65,   39,   63
        }
    };

    vector<vector<uint8_t>> ref_depth =
    {
        {
              228,  189,   30,   54,  192,   99,   17,   54,  198,  205,
               42,   54,  153,  171,   64,   54,  176,  195,   77,   54,
               66,  211,  223,   53,  111,   11,  255,   53,  106,   58,
               18,   54,  104,  198,   25,   54,  118,  205,   48,   54,
               82,   42,   10,   54,   88,   83,   15,   54,  196,  207,
                4,   54,  116,  199,    5,   54,  250,   89,   46,   54,
              145,  132,   21,   54,  180,   25,   12,   54,   87,  127,
              249,   53,   81,   86,  244,   53,  187,   58,   12,   54,
              141,   83,  139,   53,   25,   66,  157,   53,  230,  195,
              201,   53,   95,  144,  239,   53,   89,  221,  188,   53
        },
        {
                4,   63,    9,   54,  112,  194,   22,   54,  170,  235,
               23,   54,  208,   54,    8,   54
        }
    };

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

    float* const float_data = reinterpret_cast<float*>(&depth.data_[0]);
    unit_test::Rand(float_data, depth_width * depth_height, 0.0, 1.0, 0);
    unit_test::Rand(color.data_, 130, 200, 0);

    size_t num_of_levels = 2;
    auto rgbdImage = open3d::CreateRGBDImageFromColorAndDepth(color, depth);
    auto pyramid = open3d::CreateRGBDImagePyramid(*rgbdImage, num_of_levels);
    auto filteredPyramid = open3d::FilterRGBDImagePyramid(pyramid, open3d::Image::FilterType::Gaussian3);

    for (size_t j = 0; j < num_of_levels; j++)
    {
        EXPECT_EQ(ref_color[j].size(), filteredPyramid[j]->color_.data_.size());
        for (size_t i = 0; i < ref_color[j].size(); i++)
            EXPECT_EQ(ref_color[j][i], filteredPyramid[j]->color_.data_[i]);

        EXPECT_EQ(ref_depth[j].size(), filteredPyramid[j]->depth_.data_.size());
        for (size_t i = 0; i < ref_depth[j].size(); i++)
            EXPECT_EQ(ref_depth[j][i], filteredPyramid[j]->depth_.data_[i]);
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(RGBDImage, CreateRGBDImagePyramid)
{
    vector<vector<uint8_t>> ref_color =
    {
        {
              216,    2,   42,   63,   21,  162,   57,   63,   62,  210,
               42,   63,  216,   72,   38,   63,  116,   49,   38,   63,
               55,  245,   52,   63,  150,   19,   30,   63,  123,    6,
               19,   63,  193,   10,   23,   63,   83,  253,   46,   63,
              141,    0,   52,   63,    9,  144,   38,   63,  193,   19,
               55,   63,   19,  179,   45,   63,   56,  160,   49,   63,
               26,   10,   50,   63,  121,  185,   46,   63,  168,  239,
               16,   63,  183,  184,   35,   63,   63,  137,   21,   63,
              135,    1,   54,   63,  220,   15,   35,   63,  177,  246,
               44,   63,  207,   89,   38,   63,   56,   66,   57,   63
        },
        {
               96,  244,   44,   63,  151,  211,   36,   63,  137,   61,
               45,   63,   40,  111,   37,   63
        }
    };

    vector<vector<uint8_t>> ref_depth =
    {
        {
              172,  219,   92,   54,  209,  104,  206,   53,  157,   96,
               77,   54,  110,  129,   81,   54,   89,  111,  111,   54,
              209,  104,   78,   53,  178,  114,  175,   53,  204,   63,
               73,   54,  146,  124,  144,   53,  199,  132,   17,   54,
               99,  193,  249,   53,  168,   32,   37,   54,  246,  245,
              191,   53,  136,   42,    6,   54,   99,  193,  121,   54,
              141,  119,  112,   54,   16,   49,   39,   54,   37,  213,
               59,   54,   99,  157,   20,   53,  110,  239,   30,   54,
               32,   26,  132,   51,  204,  209,  123,   53,  194,   91,
               12,   53,  214,  145,   83,   54,  214,  255,   32,   53
        },
        {
              208,  177,  231,   53,    8,   24,   44,   54,  126,  106,
               46,   54,    0,  145,  227,   53
        }
    };

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

    float* const float_data = reinterpret_cast<float*>(&depth.data_[0]);
    unit_test::Rand(float_data, depth_width * depth_height, 0.0, 1.0, 0);
    unit_test::Rand(color.data_, 130, 200, 0);

    size_t num_of_levels = 2;
    auto rgbdImage = open3d::CreateRGBDImageFromColorAndDepth(color, depth);
    auto pyramid = open3d::CreateRGBDImagePyramid(*rgbdImage, num_of_levels);

    for (size_t j = 0; j < num_of_levels; j++)
    {
        EXPECT_EQ(ref_color[j].size(), pyramid[j]->color_.data_.size());
        for (size_t i = 0; i < ref_color[j].size(); i++)
            EXPECT_EQ(ref_color[j][i], pyramid[j]->color_.data_[i]);

        EXPECT_EQ(ref_depth[j].size(), pyramid[j]->depth_.data_.size());
        for (size_t i = 0; i < ref_depth[j].size(); i++)
            EXPECT_EQ(ref_depth[j][i], pyramid[j]->depth_.data_[i]);
    }
}
