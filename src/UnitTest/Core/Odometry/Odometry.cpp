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

// ----------------------------------------------------------------------------
// Create dummy correspondence map object.
// ----------------------------------------------------------------------------
shared_ptr<Image> CorrespondenceMap(const int& width,
                        const int& height,
                        const int& vmin,
                        const int& vmax,
                        const int& seed)
{
    int num_of_channels = 2;
    int bytes_per_channel = 4;

    Image image;

    image.PrepareImage(width,
                       height,
                       num_of_channels,
                       bytes_per_channel);

    int* const int_data = reinterpret_cast<int*>(&image.data_[0]);
    size_t image_size = image.data_.size() / sizeof(int);
    Rand(int_data, image_size, vmin, vmax, seed);

    return make_shared<Image>(image);
}

// ----------------------------------------------------------------------------
// Create dummy depth buffer object.
// ----------------------------------------------------------------------------
shared_ptr<Image> DepthBuffer(const int& width,
                  const int& height,
                  const float& vmin,
                  const float& vmax,
                  const int& seed)
{
    int num_of_channels = 1;
    int bytes_per_channel = 4;

    Image image;

    image.PrepareImage(width,
                       height,
                       num_of_channels,
                       bytes_per_channel);

    float* const float_data = reinterpret_cast<float*>(&image.data_[0]);
    size_t image_size = image.data_.size() / sizeof(float);
    Rand(float_data, image_size, vmin, vmax, seed);

    return make_shared<Image>(image);
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
TEST(Odometry, MergeCorrespondenceMaps)
{
    vector<int> ref_map =
    {
             4,     1,     3,     3,     4,     0,     1,     3,     0,     2,
             1,     2,     1,     2,     4,     4,     2,     3,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     4,     0,
             2,     4,     2,     0,     2,     2,     1,     4,     0,     3,
             2,     3,     1,     4,     0,     1,     3,     4,    -1,    -1
    };

    vector<float> ref_depth =
    {
           -0.980254,   -0.990773,   -0.981638,   -0.981269,   -0.978593,
           -0.995386,   -0.992157,   -0.982007,   -0.993541,   -1.000000,
           -1.000000,   -1.000000,   -1.000000,   -1.000000,   -0.977670,
           -0.978501,   -0.985052,   -0.983206,   -0.996678,   -0.985790,
           -0.999631,   -0.994371,   -0.996863,   -0.981084,   -1.000000
    };

    int width = 5;
    int height = 5;

    shared_ptr<Image> map;
    shared_ptr<Image> depth;

    tie(map, depth) = InitializeCorrespondenceMap(width, height);

    shared_ptr<Image> map_part = CorrespondenceMap(width, height, -1, 5, 0);
    shared_ptr<Image> depth_part = DepthBuffer(width, height, -1.0, 5.0, 0);

    MergeCorrespondenceMaps(*map, *depth, *map_part, *depth_part);

    size_t map_size = map->data_.size() / sizeof(int);
    int* const map_data = reinterpret_cast<int*>(&map->data_[0]);
    EXPECT_EQ(ref_map.size(), map_size);
    for (size_t i = 0; i < ref_map.size(); i++)
        EXPECT_EQ(ref_map[i], map_data[i]);

    size_t depth_size = depth->data_.size() / sizeof(float);
    float* const depth_data = reinterpret_cast<float*>(&depth->data_[0]);
    EXPECT_EQ(ref_depth.size(), depth_size);
    for (size_t i = 0; i < ref_depth.size(); i++)
        EXPECT_NEAR(ref_depth[i], depth_data[i], THRESHOLD_1E_6);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Odometry, CountCorrespondence)
{
    int width = 5;
    int height = 5;

    shared_ptr<Image> map = CorrespondenceMap(width, height, -1, 5, 0);

    int output = CountCorrespondence(*map);

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
