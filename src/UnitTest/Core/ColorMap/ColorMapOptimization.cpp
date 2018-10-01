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

#include "Core/ColorMap/ColorMapOptimization.h"
#include "Core/Camera/PinholeCameraTrajectory.h"

using namespace std;

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(ColorMapOptimization, Project3DPointAndGetUVDepth)
{
    vector<Eigen::Vector3d> ref_points =
    {
        {    1.072613,    0.611307,    0.168320 },
        {    0.897783,    0.859754,    0.171703 },
        {    1.452353,    1.769294,    0.071895 },
        {    1.181915,    0.663475,    0.119262 },
        {    1.498387,    0.741398,    0.078662 },
        {    0.814378,    0.620043,    0.197078 },
        {    1.458333,    1.693333,    0.030450 },
        {    1.709016,    2.412951,    0.051596 },
        {    1.288462,    2.510000,    0.032987 },
        {    2.316667,    1.043333,    0.022837 },
        {    1.029231,    0.366000,    0.109958 },
        {    1.390000,    0.585733,    0.063437 },
        {    0.973200,    0.512240,    0.105729 },
        {    0.948980,    0.437551,    0.165782 },
        {    1.461765,    1.644902,    0.086275 },
        {    1.535393,    1.109551,    0.075279 },
        {    3.608824,    5.121765,    0.014379 },
        {    3.350000,    4.361429,    0.017762 },
        {    0.797577,    0.636344,    0.192003 },
        {    9.990000,    8.046000,    0.004229 },
        {    0.770000,    1.511333,    0.050750 },
        {    0.834722,    0.595556,    0.182699 },
        {    0.857368,    0.744105,    0.080354 },
        {    1.111765,    0.977059,    0.143791 },
        {    0.855405,    0.429640,    0.093887 },
        {    0.917213,    0.730765,    0.154787 },
        {    0.810736,    0.506319,    0.137870 },
        {    0.942857,    3.160476,    0.035525 },
        {    1.111137,    0.389431,    0.178470 },
        {    0.822687,    0.615727,    0.192003 }
    };

    Eigen::Vector3d point = { 3.3, 4.4, 5.5 };
    open3d::PinholeCameraTrajectory camera;
    camera.extrinsic_.resize(1);

    int width = 320;
    int height = 240;

    double fx = 0.5;
    double fy = 0.65;

    double cx = 0.75;
    double cy = 0.35;

    camera.intrinsic_.SetIntrinsics(width, height, fx, fy, cx, cy);

    std::pair<double, double> f = camera.intrinsic_.GetFocalLength();
    std::pair<double, double> p = camera.intrinsic_.GetPrincipalPoint();

    for (int i = 0; i < ref_points.size(); i++)
    {
        Eigen::Matrix4d pose;
        pose << 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0;

        // change the pose randomly
        vector<double> xyz(3);
        unit_test::Rand(xyz, 0.0, 10.0, i);

        pose(0, 0) = xyz[0];
        pose(1, 1) = xyz[1];
        pose(2, 2) = xyz[2];

        camera.extrinsic_[0] = pose;

        int camid = 0;

        float u, v, d;
        tie(u, v, d) = Project3DPointAndGetUVDepth(point, camera, camid);
        unit_test::ExpectEQ(ref_points[i], Eigen::Vector3d(u, v, d));
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(ColorMapOptimization, DISABLED_MakeVertexAndImageVisibility)
{
    unit_test::NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(ColorMapOptimization, DISABLED_MakeWarpingFields)
{
    unit_test::NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(ColorMapOptimization, DISABLED_QueryImageIntensity)
{
    unit_test::NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(ColorMapOptimization, DISABLED_SetProxyIntensityForVertex)
{
    unit_test::NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(ColorMapOptimization, DISABLED_OptimizeImageCoorNonrigid)
{
    unit_test::NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(ColorMapOptimization, DISABLED_OptimizeImageCoorRigid)
{
    unit_test::NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(ColorMapOptimization, DISABLED_SetGeometryColorAverage)
{
    unit_test::NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(ColorMapOptimization, DISABLED_MakeGradientImages)
{
    unit_test::NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(ColorMapOptimization, DISABLED_MakeDepthMasks)
{
    unit_test::NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(ColorMapOptimization, DISABLED_ColorMapOptimization)
{
    unit_test::NotImplemented();
}
