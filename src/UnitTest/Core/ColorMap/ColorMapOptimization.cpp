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
        {    1.072613,    0.611307,   42.921570 },
        {    0.897783,    0.859754,   43.784313 },
        {    1.452353,    1.769294,   18.333334 },
        {    1.181915,    0.663475,   30.411764 },
        {    1.498387,    0.741398,   20.058823 },
        {    0.814378,    0.620043,   50.254902 },
        {    1.458333,    1.693333,    7.764706 },
        {    1.709016,    2.412951,   13.156863 },
        {    1.288462,    2.510000,    8.411765 },
        {    2.316667,    1.043333,    5.823529 },
        {    1.029231,    0.366000,   28.039215 },
        {    1.390000,    0.585733,   16.176470 },
        {    0.973200,    0.512240,   26.960785 },
        {    0.948980,    0.437551,   42.274509 },
        {    1.461765,    1.644902,   22.000000 },
        {    1.535393,    1.109551,   19.196079 },
        {    3.608824,    5.121765,    3.666667 },
        {    3.350000,    4.361429,    4.529412 },
        {    0.797577,    0.636344,   48.960785 },
        {    9.990000,    8.046000,    1.078431 },
        {    0.770000,    1.511333,   12.941176 },
        {    0.834722,    0.595556,   46.588234 },
        {    0.857368,    0.744105,   20.490196 },
        {    1.111765,    0.977059,   36.666668 },
        {    0.855405,    0.429640,   23.941177 },
        {    0.917213,    0.730765,   39.470589 },
        {    0.810736,    0.506319,   35.156864 },
        {    0.942857,    3.160476,    9.058824 },
        {    1.111137,    0.389431,   45.509804 },
        {    0.822687,    0.615727,   48.960785 }
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
PinholeCameraTrajectory GenerateCamera(const int& width,
                                       const int& height,
                                       const Eigen::Vector3d& pose)
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

    // not absolutely necessary but just in case
    camera.extrinsic_[0] << 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0;

    camera.extrinsic_[0](0, 0) = pose(0, 0);
    camera.extrinsic_[0](1, 1) = pose(1, 0);
    camera.extrinsic_[0](2, 2) = pose(2, 0);

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
             591,  592,  593,  670,  671,  672,  673,  674,  749,  750,
             751,  752,  753,  754,  755,  828,  829,  830,  831,  832,
             833,  834,  835,  836,  907,  908,  909,  910,  911,  912,
             913,  914,  915,  916,  917,  962,  963,  964,  965,  966,
             967,  968,  986,  987,  988,  989,  990,  991,  992,  993,
             994,  995,  996,  997,  998, 1036, 1037, 1038, 1039, 1040,
            1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050,
            1051, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074,
            1075, 1076, 1077, 1078, 1113, 1114, 1115, 1116, 1117, 1118,
            1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128,
            1129, 1130, 1131, 1132, 1133, 1134, 1145, 1146, 1147, 1148,
            1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158,
            1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199,
            1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209,
            1210, 1211, 1212, 1213, 1214, 1215, 1216, 1224, 1225, 1226,
            1227, 1228, 1229, 1230, 1231, 1232, 1270, 1271, 1272, 1273,
            1274, 1275, 1276, 1277, 1278, 1279, 1280, 1281, 1284, 1285,
            1286, 1287, 1288, 1289, 1290, 1291, 1292, 1293, 1294, 1298,
            1299, 1300, 1301, 1302, 1303, 1362, 1370, 1371, 1372, 1373,
            1374, 1375, 1380, 1381, 1382, 1437, 1438, 1439, 1440, 1441,
            1442, 1444, 1445, 1446, 1447, 1448, 1455, 1456, 1457, 1458,
            1459, 1460, 1461, 1462, 1463, 1464, 1521
        }
    };

    const int width = 320;
    const int height = 240;
    size_t size = 10;

    shared_ptr<TriangleMesh> mesh = CreateMeshSphere(1.0, 40);
    vector<RGBDImage> images_rgbd = GenerateRGBDImages(width, height, size);
    vector<Image> images_mask = GenerateImages(width, height, size);

    Eigen::Vector3d pose(0.0329104, 0.0153787, 0.0306036);
    PinholeCameraTrajectory camera = GenerateCamera(width, height, pose);

    ColorMapOptimizationOption option(false, 4, 0.316, 30, 2.5, 0.03, 0.1, 3);

    vector<vector<int>> first;
    vector<vector<int>> second;
    tie(first, second) = MakeVertexAndImageVisibility(*mesh,
                                                      images_rgbd,
                                                      images_mask,
                                                      camera,
                                                      option);

    // first is a large vector of empty vectors.
    // TODO: perhaps a different kind of initialization is necessary in order
    // to fill the first vector with data that can be used for validation
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
TEST(ColorMapOptimization, QueryImageIntensity)
{
    vector<bool> ref_bool =
    {
        0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0,
        0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0
    };

    vector<Eigen::Vector3d> ref_float =
    {
        {    0.000000,    0.000000,    0.000000 },
        {   10.260207,   10.070588,   10.323875 },
        {   10.257440,   10.114879,   10.260207 },
        {    0.000000,    0.000000,    0.000000 },
        {   10.285121,   10.244983,   10.109343 },
        {    0.000000,    0.000000,    0.000000 },
        {   10.229758,   10.322492,   10.073357 },
        {   10.300346,   10.211764,   10.112111 },
        {   10.229758,   10.113495,   10.211764 },
        {   10.261592,   10.318339,   10.346021 },
        {    0.000000,    0.000000,    0.000000 },
        {    0.000000,    0.000000,    0.000000 },
        {    0.000000,    0.000000,    0.000000 },
        {    0.000000,    0.000000,    0.000000 },
        {   10.157785,   10.347404,   10.249135 },
        {   10.343252,   10.102422,   10.271280 },
        {   10.094118,   10.066436,   10.243599 },
        {   10.024914,   10.001384,   10.325259 },
        {    0.000000,    0.000000,    0.000000 },
        {   10.073357,   10.159169,   10.055364 },
        {    0.000000,    0.000000,    0.000000 },
        {    0.000000,    0.000000,    0.000000 },
        {    0.000000,    0.000000,    0.000000 },
        {   10.217301,   10.098269,   10.276816 },
        {    0.000000,    0.000000,    0.000000 }
    };

    const int width = 320;
    const int height = 240;
    const int num_of_channels = 3;
    const int bytes_per_channel = 4;

    Image img;
    img.PrepareImage(width,
                    height,
                    num_of_channels,
                    bytes_per_channel);
    float* const depthFloatData = reinterpret_cast<float*>(&img.data_[0]);
    Rand(depthFloatData, width * height, 10.0, 100.0, 0);

    Eigen::Vector3d pose(62.5, 37.5, 1.85);
    PinholeCameraTrajectory camera = GenerateCamera(width, height, pose);
    int camid = 0;
    int ch = -1;

    size_t size = 25;

    for (size_t i = 0; i < size; i++)
    {
        vector<double> vData(3);
        Rand(vData, 10.0, 100.0, i);
        Eigen::Vector3d V(vData[0], vData[1], vData[2]);

        bool boolResult = false;
        float floatResult = 0.0;

        tie(boolResult, floatResult) = QueryImageIntensity<float>(img,
                                                                  V,
                                                                  camera,
                                                                  camid,
                                                                  0);
        EXPECT_EQ(ref_bool[i], boolResult);
        EXPECT_NEAR(ref_float[i](0, 0), floatResult, THRESHOLD_1E_6);

        tie(boolResult, floatResult) = QueryImageIntensity<float>(img,
                                                                  V,
                                                                  camera,
                                                                  camid,
                                                                  1);
        EXPECT_EQ(ref_bool[i], boolResult);
        EXPECT_NEAR(ref_float[i](1, 0), floatResult, THRESHOLD_1E_6);

        tie(boolResult, floatResult) = QueryImageIntensity<float>(img,
                                                                  V,
                                                                  camera,
                                                                  camid,
                                                                  2);
        EXPECT_EQ(ref_bool[i], boolResult);
        EXPECT_NEAR(ref_float[i](2, 0), floatResult, THRESHOLD_1E_6);
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(ColorMapOptimization, QueryImageIntensity_WarpingField)
{
    vector<bool> ref_bool =
    {
        0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0,
        0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0
    };

    vector<Eigen::Vector3d> ref_float =
    {
        {    0.000000,    0.000000,    0.000000 },
        {   10.260207,   10.070588,   10.323875 },
        {   10.257440,   10.114879,   10.260207 },
        {    0.000000,    0.000000,    0.000000 },
        {   10.285121,   10.244983,   10.109343 },
        {    0.000000,    0.000000,    0.000000 },
        {   10.229758,   10.322492,   10.073357 },
        {   10.300346,   10.211764,   10.112111 },
        {   10.229758,   10.113495,   10.211764 },
        {   10.261592,   10.318339,   10.346021 },
        {    0.000000,    0.000000,    0.000000 },
        {    0.000000,    0.000000,    0.000000 },
        {    0.000000,    0.000000,    0.000000 },
        {    0.000000,    0.000000,    0.000000 },
        {   10.157785,   10.347404,   10.249135 },
        {   10.343252,   10.102422,   10.271280 },
        {   10.094118,   10.066436,   10.243599 },
        {   10.024914,   10.001384,   10.325259 },
        {    0.000000,    0.000000,    0.000000 },
        {   10.073357,   10.159169,   10.055364 },
        {    0.000000,    0.000000,    0.000000 },
        {    0.000000,    0.000000,    0.000000 },
        {    0.000000,    0.000000,    0.000000 },
        {   10.217301,   10.098269,   10.276816 },
        {    0.000000,    0.000000,    0.000000 }
    };

    const int width = 320;
    const int height = 240;
    const int num_of_channels = 3;
    const int bytes_per_channel = 4;

    Image img;
    img.PrepareImage(width,
                    height,
                    num_of_channels,
                    bytes_per_channel);
    float* const depthFloatData = reinterpret_cast<float*>(&img.data_[0]);
    Rand(depthFloatData, width * height, 10.0, 100.0, 0);

    // TODO: initialize the field in such a way that it has an effect on the
    // outcome of QueryImageIntensity.
    const int nr_anchors = 16;
    open3d::ImageWarpingField field(width, height, nr_anchors);

    Eigen::Vector3d pose(62.5, 37.5, 1.85);
    PinholeCameraTrajectory camera = GenerateCamera(width, height, pose);
    int camid = 0;
    int ch = -1;

    size_t size = 25;

    for (size_t i = 0; i < size; i++)
    {
        vector<double> vData(3);
        Rand(vData, 10.0, 100.0, i);
        Eigen::Vector3d V(vData[0], vData[1], vData[2]);

        bool boolResult = false;
        float floatResult = 0.0;

        tie(boolResult, floatResult) = QueryImageIntensity<float>(img,
                                                                  V,
                                                                  camera,
                                                                  camid,
                                                                  0);
        EXPECT_EQ(ref_bool[i], boolResult);
        EXPECT_NEAR(ref_float[i](0, 0), floatResult, THRESHOLD_1E_6);

        tie(boolResult, floatResult) = QueryImageIntensity<float>(img,
                                                                  V,
                                                                  camera,
                                                                  camid,
                                                                  1);
        EXPECT_EQ(ref_bool[i], boolResult);
        EXPECT_NEAR(ref_float[i](1, 0), floatResult, THRESHOLD_1E_6);

        tie(boolResult, floatResult) = QueryImageIntensity<float>(img,
                                                                  V,
                                                                  camera,
                                                                  camid,
                                                                  2);
        EXPECT_EQ(ref_bool[i], boolResult);
        EXPECT_NEAR(ref_float[i](2, 0), floatResult, THRESHOLD_1E_6);
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(ColorMapOptimization, SetProxyIntensityForVertex)
{
    vector<double> ref_proxy_intensity =
    {
            0.000000,    0.000000,    0.000000,    0.000000,    0.000000,
            0.000000,    0.000000,    0.000000,    0.000000,    0.000000,
            0.000000,    0.000000,    0.000000,    0.000000,    0.000000,
            0.000000,    0.000000,    0.000000,    0.000000,    0.000000,
            0.000000,    0.000000,    0.000000,    0.000000,   10.120416,
           10.113495,   10.192388,    0.000000,    0.000000,    0.000000,
            0.000000,    0.000000,    0.000000,    0.000000,    0.000000,
            0.000000,    0.000000,    0.000000,    0.000000,    0.000000,
            0.000000,    0.000000,    0.000000,   10.244983,   10.272664,
           10.304499,   10.328028,    0.000000,    0.000000,    0.000000,
            0.000000,    0.000000,    0.000000,    0.000000,    0.000000,
            0.000000,    0.000000,    0.000000,    0.000000,    0.000000,
            0.000000,    0.000000,    0.000000,   10.262976,   10.120416,
           10.106574,   10.181314,    0.000000,    0.000000,    0.000000,
            0.000000,    0.000000,    0.000000,    0.000000,    0.000000,
            0.000000,    0.000000,    0.000000,    0.000000,    0.000000,
            0.000000,    0.000000,    0.000000,    0.000000,    0.000000,
            0.000000,    0.000000,    0.000000,    0.000000,    0.000000,
            0.000000,    0.000000,    0.000000,    0.000000,    0.000000,
            0.000000,    0.000000,    0.000000,    0.000000,    0.000000,
            0.000000,    0.000000,    0.000000,    0.000000,    0.000000,
            0.000000,    0.000000,    0.000000,    0.000000,    0.000000,
            0.000000,    0.000000,    0.000000,   10.262976,   10.120416,
           10.106574,   10.181314,    0.000000,    0.000000,    0.000000,
            0.000000,    0.000000,    0.000000,    0.000000,    0.000000,
            0.000000,    0.000000,    0.000000,    0.000000,    0.000000,
            0.000000,    0.000000,    0.000000,   10.244983,   10.272664,
           10.304499,   10.328028,    0.000000,    0.000000,    0.000000,
            0.000000,    0.000000,    0.000000,    0.000000,    0.000000,
            0.000000,    0.000000,    0.000000,    0.000000,    0.000000,
            0.000000,    0.000000,    0.000000,    0.000000,   10.120416,
           10.113495,   10.192388,    0.000000,    0.000000,    0.000000,
            0.000000,    0.000000,    0.000000,    0.000000,    0.000000,
            0.000000,    0.000000,    0.000000,    0.000000,    0.000000,
            0.000000,    0.000000,    0.000000,    0.000000,    0.000000,
            0.000000,    0.000000,    0.000000,    0.000000,    0.000000,
            0.000000,    0.000000
    };

    size_t size = 10;

    const int width = 320;
    const int height = 240;
    const int num_of_channels = 3;
    const int bytes_per_channel = 4;

    shared_ptr<TriangleMesh> mesh = CreateMeshSphere(10.0, 10);

    vector<shared_ptr<Image>> images_gray;
    for (size_t i = 0; i < size; i++)
    {
        Image image;

        image.PrepareImage(width,
                           height,
                           num_of_channels,
                           bytes_per_channel);

        float* const depthFloatData = reinterpret_cast<float*>(&image.data_[0]);
        Rand(depthFloatData, width * height, 10.0, 100.0, 0);

        images_gray.push_back(make_shared<Image>(image));
    }

    Eigen::Vector3d pose(30, 15, 0.3);
    PinholeCameraTrajectory camera = GenerateCamera(width, height, pose);
    int camid = 0;

    int n_vertex = mesh->vertices_.size();
    vector<vector<int>> visiblity_vertex_to_image(n_vertex, vector<int>(size, 0));

    vector<double> proxy_intensity;

    SetProxyIntensityForVertex(*mesh,
                               images_gray,
                               camera,
                               visiblity_vertex_to_image,
                               proxy_intensity);

    EXPECT_EQ(ref_proxy_intensity.size(), proxy_intensity.size());
    for(size_t i = 0; i < proxy_intensity.size(); i++)
        EXPECT_NEAR(ref_proxy_intensity[i],
                    proxy_intensity[i], THRESHOLD_1E_6);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(ColorMapOptimization, DISABLED_SetProxyIntensityForVertex_WarpingField)
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
