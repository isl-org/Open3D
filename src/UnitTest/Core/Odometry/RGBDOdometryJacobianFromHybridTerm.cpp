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
TEST(RGBDOdometryJacobianFromHybridTerm, ComputeJacobianAndResidual)
{
    vector<vector<double>> ref_J_r =
    {
        {   -0.000847,    0.000436,   -0.000029,    0.030973,    0.046549,   -0.208322 },
        {   -1.330339,    0.691745,   -0.071085,   54.062151,   70.280796, -327.840711 },
        {   -0.000237,    0.000098,    0.000014,    0.005430,    0.023047,   -0.070798 },
        {   -0.912094,    0.424251,   -0.074299,   57.175133,   74.327673, -277.466475 },
        {   -0.000165,    0.000086,    0.000021,    0.006665,    0.021429,   -0.034813 },
        {   -0.302894,    0.197598,   -0.017387,   25.694534,   33.402894,  -68.003781 },
        {   -0.000156,    0.000083,   -0.000013,    0.017233,    0.025900,   -0.041265 },
        {   -0.243354,    0.137160,   -0.029837,   30.079692,   39.103599,  -65.576051 },
        {   -0.000090,    0.000138,    0.000074,    0.013057,    0.026204,   -0.033402 },
        {   -0.147759,    0.279768,    0.096142,   29.240504,   38.012656,  -65.675815 },
        {   -0.000008,    0.000024,   -0.000003,    0.003427,    0.000730,   -0.004052 },
        {   -0.134877,    0.179051,    0.055068,   20.094244,   26.122517,  -35.719647 },
        {   -0.000033,    0.000034,   -0.000020,    0.007167,    0.001957,   -0.008644 },
        {   -0.303142,    0.197761,   -0.017401,   25.715819,   33.430564,  -68.059298 },
        {   -0.000237,    0.000098,    0.000014,    0.005430,    0.023047,   -0.070798 },
        {   -0.912094,    0.424251,   -0.074299,   57.175133,   74.327673, -277.466475 },
        {   -0.001459,    0.001248,    0.000135,    0.020795,    0.066858,   -0.394653 },
        {   -2.970204,    2.590453,    0.039328,   80.166945,  104.217029, -810.028693 },
        {   -0.000011,    0.000038,   -0.000002,    0.004978,    0.001061,   -0.008368 },
        {   -0.147495,    0.279272,    0.095969,   29.187988,   37.944385,  -65.559629 }
    };

    vector<double> ref_r =
    {
            0.000294,    0.983265,   -0.000253,    0.983734,    0.000193,
            0.983930,    0.000396,    0.982372,    0.000586,    0.983628,
           -0.000248,    0.980919,   -0.000382,    0.984747,   -0.000253,
            0.983734,    0.000085,    0.985534,   -0.000066,    0.981858
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
    ShiftUP(tgtColor, 5);

    ShiftLeft(dxColor, 10);
    ShiftUP(dyColor, 5);

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

    RGBDOdometryJacobianFromHybridTerm jacobian_method;

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

        EXPECT_NEAR(ref_r[2 * row + 0], r[0], THRESHOLD_1E_6);
        EXPECT_NEAR(ref_r[2 * row + 1], r[1], THRESHOLD_1E_6);

        for (int i = 0; i < 6; i++)
        {
            EXPECT_NEAR(ref_J_r[2 * row + 0][i], J_r[0](i, 0), THRESHOLD_1E_6);
            EXPECT_NEAR(ref_J_r[2 * row + 1][i], J_r[1](i, 0), THRESHOLD_1E_6);
        }
    }
}
