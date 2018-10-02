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
#include "Core/Geometry/Image.h"
#include "Core/Geometry/RGBDImage.h"
#include "Core/Geometry/TriangleMesh.h"

using namespace open3d;
using namespace std;
using namespace unit_test;

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
    PinholeCameraTrajectory camera;
    camera.extrinsic_.resize(1);

    int width = 320;
    int height = 240;

    double fx = 0.5;
    double fy = 0.65;

    double cx = 0.75;
    double cy = 0.35;

    camera.intrinsic_.SetIntrinsics(width, height, fx, fy, cx, cy);

    pair<double, double> f = camera.intrinsic_.GetFocalLength();
    pair<double, double> p = camera.intrinsic_.GetPrincipalPoint();

    for (int i = 0; i < ref_points.size(); i++)
    {
        Eigen::Matrix4d pose;
        pose << 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0;

        // change the pose randomly
        vector<double> xyz(3);
        Rand(xyz, 0.0, 10.0, i);

        pose(0, 0) = xyz[0];
        pose(1, 1) = xyz[1];
        pose(2, 2) = xyz[2];

        camera.extrinsic_[0] = pose;

        int camid = 0;

        float u, v, d;
        tie(u, v, d) = Project3DPointAndGetUVDepth(point, camera, camid);
        ExpectEQ(ref_points[i], Eigen::Vector3d(u, v, d));
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(ColorMapOptimization, MakeVertexAndImageVisibility)
{
    size_t size = 10;

    // test image dimensions
    const int width = 5;
    const int height = 5;
    const int num_of_channels = 3;
    const int bytes_per_channel = 1;
    const int depth_num_of_channels = 1;
    const int depth_bytes_per_channel = 4;

    // generate triangle mesh
    int nrVertices = 100;
    int nrTriangles = 30;

    Eigen::Vector3d dmin(0.0, 0.0, 0.0);
    Eigen::Vector3d dmax(10.0, 10.0, 10.0);

    Eigen::Vector3i imin(0, 0, 0);
    Eigen::Vector3i imax(nrVertices - 1, nrVertices - 1, nrVertices - 1);

    TriangleMesh mesh;

    mesh.vertices_.resize(nrVertices);
    mesh.vertex_normals_.resize(nrVertices);
    mesh.vertex_colors_.resize(nrVertices);
    mesh.triangles_.resize(nrTriangles);
    mesh.triangle_normals_.resize(nrTriangles);

    unit_test::Rand(mesh.vertices_,         dmin, dmax, 0);
    unit_test::Rand(mesh.vertex_normals_,   dmin, dmax, 0);
    unit_test::Rand(mesh.vertex_colors_,    dmin, dmax, 0);
    unit_test::Rand(mesh.triangles_,        imin, imax, 0);
    unit_test::Rand(mesh.triangle_normals_, dmin, dmax, 0);

    // generate input RGBD images
    vector<RGBDImage> images_rgbd;
    for (size_t i = 0; i < size; i++)
    {
        Image color;
        Image depth;

        color.PrepareImage(width,
                           height,
                           num_of_channels,
                           bytes_per_channel);

        depth.PrepareImage(width,
                           height,
                           depth_num_of_channels,
                           depth_bytes_per_channel);

        Rand(color.data_, 0, 255, i);

        float* const depthData = reinterpret_cast<float*>(&depth.data_[0]);
        Rand(depthData, width * height, 0.0f, 10.0f, i + 1 * size);

        RGBDImage rgbdImage(color, depth);
        images_rgbd.push_back(rgbdImage);
    }

    // generate input images
    vector<Image> images_mask;
    for (size_t i = 0; i < size; i++)
    {
        Image image;

        image.PrepareImage(width,
                           height,
                           num_of_channels,
                           bytes_per_channel);

        Rand(image.data_, 0, 255, i);

        images_mask.push_back(image);
    }

    // get a camera
    PinholeCameraTrajectory camera;
    camera.extrinsic_.resize(1);

    int camera_width = 320;
    int camera_height = 240;

    double fx = 0.5;
    double fy = 0.65;

    double cx = 0.75;
    double cy = 0.35;

    camera.intrinsic_.SetIntrinsics(camera_width, camera_height, fx, fy, cx, cy);

    pair<double, double> f = camera.intrinsic_.GetFocalLength();
    pair<double, double> p = camera.intrinsic_.GetPrincipalPoint();

    Eigen::Matrix4d pose;
    pose << 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0;

    // change the pose randomly
    vector<double> xyz(3);
    Rand(xyz, 0.0, 10.0, 0);

    pose(0, 0) = xyz[0];
    pose(1, 1) = xyz[1];
    pose(2, 2) = xyz[2];

    camera.extrinsic_[0] = pose;

    ColorMapOptimizationOption option(false, 4, 0.316, 30, 2.5, 0.03, 0.1, 3);

    vector<vector<int>> first;
    vector<vector<int>> second;
    tie(first, second) = MakeVertexAndImageVisibility(mesh,
                                                      images_rgbd,
                                                      images_mask,
                                                      camera,
                                                      option);

    cout << "1st size: " << first.size() << endl;
    for (size_t i = 0; i < first.size(); i++)
        cout << "    loop " << i << " size: " << first[i].size() << endl;
    cout << endl;

    cout << "2nd size: " << second.size() << endl;
    for (size_t i = 0; i < second.size(); i++)
        cout << "    loop " << i << " size: " << second[i].size() << endl;
    cout << endl;
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(ColorMapOptimization, MakeWarpingFields)
{
    int ref_anchor_w = 4;
    int ref_anchor_h = 4;
    double ref_anchor_step = 1.666667;
    vector<double> ref_flow =
    {
            0.000000,    0.000000,    1.666667,    0.000000,    3.333333,
            0.000000,    5.000000,    0.000000,    0.000000,    1.666667,
            1.666667,    1.666667,    3.333333,    1.666667,    5.000000,
            1.666667,    0.000000,    3.333333,    1.666667,    3.333333,
            3.333333,    3.333333,    5.000000,    3.333333,    0.000000,
            5.000000,    1.666667,    5.000000,    3.333333,    5.000000,
            5.000000,    5.000000
    };

    size_t size = 10;
    vector<shared_ptr<Image>> images;

    for (size_t i = 0; i < size; i++)
    {
        Image image;

        // test image dimensions
        const int width = 5;
        const int height = 5;
        const int num_of_channels = 1;
        const int bytes_per_channel = 2;

        image.PrepareImage(width,
                           height,
                           num_of_channels,
                           bytes_per_channel);

        Rand(image.data_, 0, 255, i);

        images.push_back(make_shared<Image>(image));
    }

    ColorMapOptimizationOption option(false, 4, 0.316, 30, 2.5, 0.03, 0.1, 3);

    vector<ImageWarpingField> fields = MakeWarpingFields(images, option);

    for (size_t i = 0; i < fields.size(); i++)
    {
        EXPECT_EQ(ref_anchor_w, fields[i].anchor_w_);
        EXPECT_EQ(ref_anchor_h, fields[i].anchor_h_);
        EXPECT_NEAR(ref_anchor_step, fields[i].anchor_step_, THRESHOLD_1E_6);

        EXPECT_EQ(ref_flow.size(), fields[i].flow_.size());
        for (size_t j = 0; j < fields[i].flow_.size(); j++)
            EXPECT_NEAR(ref_flow[j], fields[i].flow_[j], THRESHOLD_1E_6);
    }
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
