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
#include "OdometryTools.h"

#include "Core/Geometry/RGBDImage.h"
#include "Core/Odometry/RGBDOdometryJacobian.h"

using namespace odometry_tools;
using namespace open3d;
using namespace std;
using namespace unit_test;

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(RGBDOdometryJacobianFromColorTerm, ComputeJacobianAndResidual)
{
    vector<vector<double>> ref_J_r =
    {
        { -0.004738, 0.002436, -0.000160, 0.173142, 0.260220, -1.164557 },
        { -0.001326, 0.000550,  0.000077, 0.030357, 0.128839, -0.395772 },
        { -0.000925, 0.000478,  0.000117, 0.037260, 0.119792, -0.194611 },
        { -0.000871, 0.000463, -0.000073, 0.096335, 0.144784, -0.230677 },
        { -0.000501, 0.000774,  0.000411, 0.072993, 0.146487, -0.186723 },
        { -0.000047, 0.000133, -0.000016, 0.019158, 0.004083, -0.022654 },
        { -0.000185, 0.000193, -0.000109, 0.040064, 0.010938, -0.048321 },
        { -0.001326, 0.000550,  0.000077, 0.030357, 0.128839, -0.395772 },
        { -0.008159, 0.006977,  0.000752, 0.116250, 0.373750, -2.206175 },
        { -0.000061, 0.000214, -0.000009, 0.027828, 0.005931, -0.046776 }
    };

    vector<double> ref_r =
    {
            0.001646,   -0.001415,    0.001077,    0.002215,    0.003276,
           -0.001384,   -0.002138,   -0.001415,    0.000477,   -0.000369
    };

    int width = 10;
    int height = 10;
    int num_of_channels = 1;
    int bytes_per_channel = 4;

    shared_ptr<Image> srcColor = GenerateImage(width, height, 1, 4, 0.0f, 1.0f, 1);
    shared_ptr<Image> srcDepth = GenerateImage(width, height, 1, 4, 0.0f, 1.0f, 0);

    shared_ptr<Image> tgtColor = GenerateImage(width, height, 1, 4, 0.0f, 1.0f, 1);
    shared_ptr<Image> tgtDepth = GenerateImage(width, height, 1, 4, 1.0f, 2.0f, 0);

    shared_ptr<Image> dxColor = GenerateImage(width, height, 1, 4, 0.0f, 1.0f, 1);
    shared_ptr<Image> dyColor = GenerateImage(width, height, 1, 4, 0.0f, 1.0f, 1);

    ShiftLeft(tgtColor, 10);
    ShiftUp(tgtColor, 5);

    ShiftLeft(dxColor, 10);
    ShiftUp(dyColor, 5);

    RGBDImage source(*srcColor, *srcDepth);
    RGBDImage target(*tgtColor, *tgtDepth);
    shared_ptr<Image> source_xyz = GenerateImage(width, height, 3, 4, 0.0f, 1.0f, 0);;
    RGBDImage target_dx(*dxColor, *tgtDepth);
    RGBDImage target_dy(*dyColor, *tgtDepth);

    Eigen::Matrix3d intrinsic = Eigen::Matrix3d::Zero();
    intrinsic(0, 0) = 0.5;
    intrinsic(1, 1) = 0.65;
    intrinsic(0, 2) = 0.75;
    intrinsic(1, 2) = 0.35;

    Eigen::Matrix4d extrinsic = Eigen::Matrix4d::Zero();
    extrinsic(0, 0) = 1.0;
    extrinsic(1, 1) = 1.0;
    extrinsic(2, 2) = 1.0;

    int rows = height;
    vector<Eigen::Vector4i> corresps(rows);
    Rand(corresps, 0, 3, 0);

    RGBDOdometryJacobianFromColorTerm jacobian_method;

    for (int row = 0; row < rows; row++)
    {
        vector<Eigen::Vector6d> J_r;
        vector<double> r;

        jacobian_method.ComputeJacobianAndResidual(
            row,
            J_r,
            r,
            source,
            target,
            *source_xyz,
            target_dx,
            target_dy,
            intrinsic,
            extrinsic,
            corresps
        );

        EXPECT_NEAR(ref_r[row], r[0], THRESHOLD_1E_6);

        for (int i = 0; i < 6; i++)
            EXPECT_NEAR(ref_J_r[row][i], J_r[0](i, 0), THRESHOLD_1E_6);
    }
}
