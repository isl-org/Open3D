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
TEST(Odometry, DISABLED_AddElementToCorrespondenceMap)
{
    unit_test::NotImplemented();
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
