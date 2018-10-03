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
vector<RGBDImage> GenerateRGBDImages(const int& width,
                                     const int& height,
                                     const size_t& size)
{
    const int num_of_channels = 3;
    const int bytes_per_channel = 1;
    const int depth_num_of_channels = 1;
    const int depth_bytes_per_channel = 4;

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

        Rand(depth.data_, 0, 255, i);

        RGBDImage rgbdImage(color, depth);
        images_rgbd.push_back(rgbdImage);
    }

    return move(images_rgbd);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
vector<Image> GenerateImages(const int& width,
                             const int& height,
                             const size_t& size)
{
    const int num_of_channels = 3;
    const int bytes_per_channel = 1;

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

    return images_mask;
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
PinholeCameraTrajectory GenerateCamera(const int& width, const int& height)
{
    PinholeCameraTrajectory camera;
    camera.extrinsic_.resize(1);

    double fx = 0.5;
    double fy = 0.65;

    double cx = 0.75;
    double cy = 0.35;

    camera.intrinsic_.SetIntrinsics(width, height, fx, fy, cx, cy);

    pair<double, double> f = camera.intrinsic_.GetFocalLength();
    pair<double, double> p = camera.intrinsic_.GetPrincipalPoint();

    Eigen::Matrix4d pose;
    pose << 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0;

    // generate a random pose
    vector<double> xyz(3);
    Rand(xyz, 0.0, 10.0, 0);

    pose(0, 0) = xyz[0];
    pose(1, 1) = xyz[1];
    pose(2, 2) = xyz[2];

    camera.extrinsic_[0] = pose;

    return camera;
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(ColorMapOptimization, MakeVertexAndImageVisibility)
{
    vector<vector<int>> ref_second =
    {
        {
            591,   592,   593,   670,   671,   672,   673,   674,   749,   750,
            751,   752,   753,   754,   755,   828,   829,   830,   831,   832,
            833,   834,   835,   836,   907,   908,   909,   910,   911,   912,
            913,   914,   915,   916,   917,   962,   963,   964,   965,   966,
            967,   968,   986,   987,   988,   989,   990,   991,   992,   993,
            994,   995,   996,   997,   998,  1036,  1037,  1038,  1039,  1040,
            1041,  1042,  1043,  1044,  1045,  1046,  1047,  1048,  1049,  1050,
            1051,  1066,  1067,  1068,  1069,  1070,  1071,  1072,  1073,  1074,
            1075,  1076,  1077,  1078,  1113,  1114,  1115,  1116,  1117,  1118,
            1119,  1120,  1121,  1122,  1123,  1124,  1125,  1126,  1127,  1128,
            1129,  1130,  1131,  1132,  1133,  1134,  1145,  1146,  1147,  1148,
            1149,  1150,  1151,  1152,  1153,  1154,  1155,  1156,  1157,  1158,
            1190,  1191,  1192,  1193,  1194,  1195,  1196,  1197,  1198,  1199,
            1200,  1201,  1202,  1203,  1204,  1205,  1206,  1207,  1208,  1209,
            1210,  1211,  1212,  1213,  1214,  1215,  1216,  1224,  1225,  1226,
            1227,  1228,  1229,  1230,  1231,  1232,  1270,  1271,  1272,  1273,
            1274,  1275,  1276,  1277,  1278,  1279,  1280,  1281,  1284,  1285,
            1286,  1287,  1288,  1289,  1290,  1291,  1292,  1293,  1294,  1298,
            1299,  1300,  1301,  1302,  1303,  1362,  1370,  1371,  1372,  1373,
            1374,  1375,  1380,  1381,  1382,  1437,  1438,  1439,  1440,  1441,
            1442,  1444,  1445,  1446,  1447,  1448,  1455,  1456,  1457,  1458,
            1459,  1460,  1461,  1462,  1463,  1464,  1521
        }
    };

    const int width = 320;
    const int height = 240;
    size_t size = 10;

    shared_ptr<TriangleMesh> mesh = CreateMeshSphere(1.0, 40);
    vector<RGBDImage> images_rgbd = GenerateRGBDImages(width, height, size);
    vector<Image> images_mask = GenerateImages(width, height, size);
    PinholeCameraTrajectory camera = GenerateCamera(width, height);
    ColorMapOptimizationOption option(false, 4, 0.316, 30, 2.5, 0.03, 0.1, 3);

    vector<vector<int>> first;
    vector<vector<int>> second;
    tie(first, second) = MakeVertexAndImageVisibility(*mesh,
                                                      images_rgbd,
                                                      images_mask,
                                                      camera,
                                                      option);

    // first is a large vector of empty vectors.
    // TODO: perhaps a different kind of initialization is necessary in order to
    // fill the first vector with data that can be used for validation
    EXPECT_EQ(3122, first.size());

    EXPECT_EQ(ref_second.size(), second.size());
    EXPECT_EQ(ref_second[0].size(), second[0].size());
    for (size_t i = 0; i < ref_second[0].size(); i++)
        EXPECT_EQ(ref_second[0][i], second[0][i]);
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
