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

#include "Core/Geometry/Image.h"
#include "Core/Geometry/RGBDImage.h"
#include "Core/Odometry/RGBDOdometryJacobian.h"

using namespace open3d;
using namespace std;
using namespace unit_test;

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
shared_ptr<Image> GenerateImage(const int width,
                                const int height,
                                const int num_of_channels,
                                const int bytes_per_channel,
                                const float& vmin,
                                const float& vmax,
                                const int& seed)
{
    Image image;

    image.PrepareImage(width,
                       height,
                       num_of_channels,
                       bytes_per_channel);

    float* const depthData = reinterpret_cast<float*>(&image.data_[0]);
    Rand(depthData, width * height, vmin, vmax, seed);

    return make_shared<Image>(image);
}

// ----------------------------------------------------------------------------
// Shift the pixels left with a specified step.
// ----------------------------------------------------------------------------
void ShiftLeft(shared_ptr<Image> image, const int& step)
{
    int width = image->width_;
    int height = image->height_;
    int num_of_channels = image->num_of_channels_;
    int bytes_per_channel = image->bytes_per_channel_;

    float* const floatData = reinterpret_cast<float*>(&image->data_[0]);
    for (int h = 0; h < height; h++)
        for (int w = 0; w < width; w++)
            floatData[h * width + w] = floatData[h * width + (w + step) % width];
}

// ----------------------------------------------------------------------------
// Shift the pixels up with a specified step.
// ----------------------------------------------------------------------------
void ShiftUP(shared_ptr<Image> image, const int& step)
{
    int width = image->width_;
    int height = image->height_;
    int num_of_channels = image->num_of_channels_;
    int bytes_per_channel = image->bytes_per_channel_;

    float* const floatData = reinterpret_cast<float*>(&image->data_[0]);
    for (int h = 0; h < height; h++)
        for (int w = 0; w < width; w++)
            floatData[h * width + w] = floatData[((h + step) % height) * width + w];
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(RGBDOdometryJacobianFromColorTerm, ComputeJacobianAndResidual)
{
    int width = 10;
    int height = 10;
    int num_of_channels = 1;
    int bytes_per_channel = 4;

    int row = 0;
    vector<Eigen::Vector6d> J_r;
    vector<double> r;

    shared_ptr<Image> srcColor = GenerateImage(width, height, 1, 4, 0.0f, 1.0f, 1);
    shared_ptr<Image> srcDepth = GenerateImage(width, height, 1, 4, 0.0f, 1.0f, 0);

    shared_ptr<Image> tgtColor = GenerateImage(width, height, 1, 4, 0.0f, 1.0f, 1);
    shared_ptr<Image> tgtDepth = GenerateImage(width, height, 1, 4, 1.0f, 2.0f, 0);

    ShiftLeft(tgtColor, 10);
    ShiftUP(tgtColor, 5);

    RGBDImage source(*srcColor, *srcDepth);
    RGBDImage target(*tgtColor, *tgtDepth);
    shared_ptr<Image> source_xyz = GenerateImage(width, height, 3, 4, 0.0f, 1.0f, 0);;
    RGBDImage target_dx(*tgtColor, *tgtDepth);
    RGBDImage target_dy(*tgtColor, *tgtDepth);

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
    for (size_t i = 0; i < corresps.size(); i++)
    {
        corresps[i](0) = 0;
        corresps[i](1) = 1;
        corresps[i](2) = 2;
        corresps[i](3) = 3;
    }

    RGBDOdometryJacobianFromHybridTerm jacobian_method;

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

    cout << row << endl;
    Print(J_r);
    Print(r);
}
