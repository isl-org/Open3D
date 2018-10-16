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

#include <Core/Odometry/Odometry.h>

using namespace open3d;
using namespace std;
using namespace unit_test;

Image CorrespondenceMap(const int& width,
                        const int& height,
                        const int& num_of_channels,
                        const int& bytes_per_channel)
{
    Image image;

    image.PrepareImage(width,
                        height,
                        num_of_channels,
                        bytes_per_channel);

    int* const intData = reinterpret_cast<int*>(&image.data_[0]);
    size_t image_size = width * height * num_of_channels * bytes_per_channel / sizeof(int);
    Rand(intData, image_size, -1, 5, 0);

    return image;
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Odometry, InitializeCorrespondenceMap)
{
    int width = 5;
    int height = 5;

    shared_ptr<Image> map;
    shared_ptr<Image> depth;

    tie(map, depth) = InitializeCorrespondenceMap(width, height);

    int* const map_data = reinterpret_cast<int*>(&map->data_[0]);
    size_t map_size = map->data_.size() / sizeof(int);
    for (int i = 0; i < map_size; i++)
        EXPECT_EQ(-1, map_data[i]);

    float* const depth_data = reinterpret_cast<float*>(&depth->data_[0]);
    size_t depth_size = depth->data_.size() / sizeof(float);
    for (int i = 0; i < depth_size; i++)
        EXPECT_NEAR(-1.0f, depth_data[i], THRESHOLD_1E_6);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Odometry, AddElementToCorrespondenceMap)
{
    vector<uint8_t> ref_map =
    {
          255,  255,  255,  255,  255,  255,  255,  255,    2,    0,
            0,    0,    1,    0,    0,    0,    4,    0,    0,    0,
            2,    0,    0,    0,    3,    0,    0,    0,    2,    0,
            0,    0,  255,  255,  255,  255,  255,  255,  255,  255,
            0,    0,    0,    0,    1,    0,    0,    0,    4,    0,
            0,    0,    1,    0,    0,    0,  255,  255,  255,  255,
          255,  255,  255,  255,    2,    0,    0,    0,    2,    0,
            0,    0,  255,  255,  255,  255,  255,  255,  255,  255,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    3,    0,    0,    0,    4,    0,    0,    0,
            1,    0,    0,    0,    1,    0,    0,    0,    4,    0,
            0,    0,    0,    0,    0,    0,    2,    0,    0,    0,
          255,  255,  255,  255,  255,  255,  255,  255,  255,  255,
          255,  255,  255,  255,  255,  255,    1,    0,    0,    0,
            4,    0,    0,    0,    2,    0,    0,    0,    3,    0,
            0,    0,    1,    0,    0,    0,    2,    0,    0,    0,
          255,  255,  255,  255,  255,  255,  255,  255,    1,    0,
            0,    0,    0,    0,    0,    0,  255,  255,  255,  255,
          255,  255,  255,  255,    0,    0,    0,    0,    3,    0,
            0,    0,  255,  255,  255,  255,  255,  255,  255,  255
    };

    vector<uint8_t> ref_depth =
    {
            0,    0,  128,  191,  243,  246,  123,   60,  235,  198,
          163,   60,   92,  146,  201,   60,    0,    0,  128,  191,
          243,  246,  251,   59,  190,   83,  106,   60,    0,    0,
          128,  191,  175,   73,  229,   60,    0,    0,  128,  191,
           32,  106,  181,   59,  130,   43,   86,   60,  148,  162,
          177,   60,  111,  180,  250,   60,   20,  205,    6,   61,
            0,    0,  128,  191,    0,    0,  128,  191,   73,  112,
          152,   60,   96,  170,  245,   60,   73,  112,   24,   61,
            0,    0,  128,  191,   17,   96,   48,   60,    0,    0,
          128,  191,   69,    3,  194,   60,    0,    0,  128,  191
    };

    size_t size = 20;
    int width = 5;
    int height = 5;

    shared_ptr<Image> map;
    shared_ptr<Image> depth;

    tie(map, depth) = InitializeCorrespondenceMap(width, height);

    vector<int> u_s(size);
    vector<int> v_s(size);
    vector<int> u_t(size);
    vector<int> v_t(size);
    vector<float> transformed_d_t(size);

    Rand(u_s, 0, 5, 0);
    Rand(v_s, 0, 5, 10);
    Rand(u_t, 0, 5, 20);
    Rand(v_t, 0, 5, 30);
    Rand(transformed_d_t, 0.0, 10.0, 0);

    for (size_t i = 0; i < size; i++)
        AddElementToCorrespondenceMap(*map,
                                      *depth,
                                      u_s[i],
                                      v_s[i],
                                      u_t[i],
                                      v_t[i],
                                      transformed_d_t[i]);

    EXPECT_EQ(ref_map.size(), map->data_.size());
    for (size_t i = 0; i < ref_map.size(); i++)
        EXPECT_EQ(ref_map[i], map->data_[i]);

    EXPECT_EQ(ref_depth.size(), depth->data_.size());
    for (size_t i = 0; i < ref_depth.size(); i++)
        EXPECT_EQ(ref_depth[i], depth->data_[i]);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Odometry, DISABLED_MergeCorrespondenceMaps)
{
    vector<uint8_t> ref_map =
    {
           91,   69,   89,   89,   95,   59,   66,   88,   63,   77,
           73,   81,   68,   75,   97,   95,   81,   85,   57,   80,
           50,   61,   56,   90,   57,   70,   56,   55,   99,   60,
           75,   91,   80,   64,   81,   76,   74,   98,   64,   88,
           76,   88,   70,   94,   64,   67,   90,   95,   53,   97,
           76,   54,   59,   83,   94,   67,   53,   50,   72,   53,
           61,   98,   95,   92,   63,   76,   68,   87,   75,   83,
           76,   51,   71,   96,   96,   85,   64,   86,   81,   67,
           84,   58,   71,   93,   91,   66,   61,   94,   67,   84,
           97,   79,   82,   92,   71,   96,   69,   90,   84,   95,
           74,   60,   97,   95,   57,   93,   81,   71,   80,   63,
           89,   65,   72,   61,   59,   63,   77,   70,   58,   95,
           55,   56,   74,   87,   99,   96,   84,   69,   87,   68,
           64,   61,   79,   62,   57,   86,   56,   89,   58,   87,
           53,   97,   52,   75,   58,   61,   89,   86,   82,   98,
           81,   87,   54,   56,   75,   53,   53,   60,   72,   90,
           78,   87,   52,   57,   99,   60,   94,   56,   99,   52,
           93,   53,   50,   96,   79,   58,   58,   69,   95,   90,
           67,   77,   78,   72,   84,   54,   76,   87,   65,   99,
           78,   93,   87,   81,   51,   87,   91,   96,   93,   91
    };

    vector<uint8_t> ref_depth =
    {
           52,   27,   34,   65,  247,  251,   32,   65,  105,  245,
           33,   65,  125,  255,   33,   65,  143,   72,   34,   65,
          251,  125,   32,   65,   44,  214,   32,   65,   85,  235,
           33,   65,   96,  176,   32,   65,   69,   99,   33,   65,
          225,   48,   33,   65,   37,  147,   33,   65,   84,  234,
           32,   65,  142,   71,   33,   65,  193,   97,   34,   65,
           20,   75,   34,   65,   47,  152,   33,   65,  147,  202,
           33,   65,  181,   90,   32,   65,    7,  132,   33,   65,
           20,   10,   32,   65,  179,  153,   32,   65,  171,   85,
           32,   65,  135,    4,   34,   65,   68,   98,   32,   65
    };

    int width = 5;
    int height = 5;
    int num_of_channels = 2;
    int bytes_per_channel = 4;

    shared_ptr<Image> map;
    shared_ptr<Image> depth;

    // shared_ptr<Image> map_part;
    shared_ptr<Image> depth_part;

    tie(map, depth) = InitializeCorrespondenceMap(width, height);
    // tie(map_part, depth_part) = InitializeCorrespondenceMap(width, height);

    Image map_part;

    map_part.PrepareImage(width,
                        height,
                        num_of_channels,
                        bytes_per_channel);

    int* const intData = reinterpret_cast<int*>(&map_part.data_[0]);
    size_t map_part_size = width * height * num_of_channels * bytes_per_channel / sizeof(int);
    Rand(intData, map_part_size, -1, 5, 0);

    MergeCorrespondenceMaps(*map, *depth, map_part, *depth_part);

    Print(map->data_);
    Print(depth->data_);

    EXPECT_EQ(ref_map.size(), map->data_.size());
    for (size_t i = 0; i < ref_map.size(); i++)
        EXPECT_EQ(ref_map[i], map->data_[i]);

    EXPECT_EQ(ref_depth.size(), depth->data_.size());
    for (size_t i = 0; i < ref_depth.size(); i++)
        EXPECT_EQ(ref_depth[i], depth->data_[i]);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Odometry, CountCorrespondence)
{
    int width = 5;
    int height = 5;
    int num_of_channels = 2;
    int bytes_per_channel = 4;

    Image image;

    image.PrepareImage(width,
                        height,
                        num_of_channels,
                        bytes_per_channel);

    int* const intData = reinterpret_cast<int*>(&image.data_[0]);
    size_t image_size = width * height * num_of_channels * bytes_per_channel / sizeof(int);
    Rand(intData, image_size, -1, 5, 0);

    int output = CountCorrespondence(image);

    EXPECT_EQ(19, output);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Odometry, DISABLED_ComputeCorrespondence)
{
    unit_test::NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Odometry, DISABLED_ConvertDepthImageToXYZImage)
{
    unit_test::NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Odometry, DISABLED_CreateCameraMatrixPyramid)
{
    unit_test::NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Odometry, DISABLED_CreateInformationMatrix)
{
    unit_test::NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Odometry, DISABLED_NormalizeIntensity)
{
    unit_test::NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Odometry, DISABLED_PackRGBDImage)
{
    unit_test::NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Odometry, DISABLED_PreprocessDepth)
{
    unit_test::NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Odometry, DISABLED_CheckImagePair)
{
    unit_test::NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Odometry, DISABLED_CheckRGBDImagePair)
{
    unit_test::NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Odometry, DISABLED_InitializeRGBDOdometry)
{
    unit_test::NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Odometry, DISABLED_DoSingleIteration)
{
    unit_test::NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Odometry, DISABLED_ComputeMultiscale)
{
    unit_test::NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Odometry, DISABLED_ComputeRGBDOdometry)
{
    unit_test::NotImplemented();
}
