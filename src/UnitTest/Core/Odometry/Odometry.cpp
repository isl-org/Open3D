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
//
// ----------------------------------------------------------------------------
TEST(Odometry, InitializeCorrespondenceMap)
{
    vector<uint8_t> ref_image0 =
    {
          255,  255,  255,  255,  255,  255,  255,  255,  255,  255,
          255,  255,  255,  255,  255,  255,  255,  255,  255,  255,
          255,  255,  255,  255,  255,  255,  255,  255,  255,  255,
          255,  255,  255,  255,  255,  255,  255,  255,  255,  255,
          255,  255,  255,  255,  255,  255,  255,  255,  255,  255,
          255,  255,  255,  255,  255,  255,  255,  255,  255,  255,
          255,  255,  255,  255,  255,  255,  255,  255,  255,  255,
          255,  255,  255,  255,  255,  255,  255,  255,  255,  255,
          255,  255,  255,  255,  255,  255,  255,  255,  255,  255,
          255,  255,  255,  255,  255,  255,  255,  255,  255,  255,
          255,  255,  255,  255,  255,  255,  255,  255,  255,  255,
          255,  255,  255,  255,  255,  255,  255,  255,  255,  255,
          255,  255,  255,  255,  255,  255,  255,  255,  255,  255,
          255,  255,  255,  255,  255,  255,  255,  255,  255,  255,
          255,  255,  255,  255,  255,  255,  255,  255,  255,  255,
          255,  255,  255,  255,  255,  255,  255,  255,  255,  255,
          255,  255,  255,  255,  255,  255,  255,  255,  255,  255,
          255,  255,  255,  255,  255,  255,  255,  255,  255,  255,
          255,  255,  255,  255,  255,  255,  255,  255,  255,  255,
          255,  255,  255,  255,  255,  255,  255,  255,  255,  255
    };

    vector<uint8_t> ref_image1 =
    {
            0,    0,  128,  191,    0,    0,  128,  191,    0,    0,
          128,  191,    0,    0,  128,  191,    0,    0,  128,  191,
            0,    0,  128,  191,    0,    0,  128,  191,    0,    0,
          128,  191,    0,    0,  128,  191,    0,    0,  128,  191,
            0,    0,  128,  191,    0,    0,  128,  191,    0,    0,
          128,  191,    0,    0,  128,  191,    0,    0,  128,  191,
            0,    0,  128,  191,    0,    0,  128,  191,    0,    0,
          128,  191,    0,    0,  128,  191,    0,    0,  128,  191,
            0,    0,  128,  191,    0,    0,  128,  191,    0,    0,
          128,  191,    0,    0,  128,  191,    0,    0,  128,  191
    };

    int width = 5;
    int height = 5;

    shared_ptr<Image> image0;
    shared_ptr<Image> image1;

    tie(image0, image1) = InitializeCorrespondenceMap(width, height);

    EXPECT_EQ(ref_image0.size(), image0->data_.size());
    for (size_t i = 0; i < ref_image0.size(); i++)
        EXPECT_EQ(ref_image0[i], image0->data_[i]);

    EXPECT_EQ(ref_image1.size(), image1->data_.size());
    for (size_t i = 0; i < ref_image1.size(); i++)
        EXPECT_EQ(ref_image1[i], image1->data_[i]);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Odometry, AddElementToCorrespondenceMap)
{
    vector<uint8_t> ref_image0 =
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

    vector<uint8_t> ref_image1 =
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

    shared_ptr<Image> image0;
    shared_ptr<Image> image1;

    tie(image0, image1) = InitializeCorrespondenceMap(width, height);

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
        AddElementToCorrespondenceMap(*image0,
                                      *image1,
                                      u_s[i],
                                      v_s[i],
                                      u_t[i],
                                      v_t[i],
                                      transformed_d_t[i]);

    EXPECT_EQ(ref_image0.size(), image0->data_.size());
    for (size_t i = 0; i < ref_image0.size(); i++)
        EXPECT_EQ(ref_image0[i], image0->data_[i]);

    EXPECT_EQ(ref_image1.size(), image1->data_.size());
    for (size_t i = 0; i < ref_image1.size(); i++)
        EXPECT_EQ(ref_image1[i], image1->data_[i]);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Odometry, DISABLED_MergeCorrespondenceMaps)
{
    unit_test::NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Odometry, DISABLED_CountCorrespondence)
{
    unit_test::NotImplemented();
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
