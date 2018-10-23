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

#include "Core/Odometry/Odometry.h"
#include "Core/Odometry/RGBDOdometryJacobian.h"

using namespace odometry_tools;
using namespace open3d;
using namespace std;
using namespace unit_test;

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
    vector<int> ref_map =
    {
            -1,    -1,     2,     1,     4,     2,     3,     2,    -1,    -1,
             0,     1,     4,     1,    -1,    -1,     2,     2,    -1,    -1,
             0,     0,     0,     3,     4,     1,     1,     4,     0,     2,
            -1,    -1,    -1,    -1,     1,     4,     2,     3,     1,     2,
            -1,    -1,     1,     0,    -1,    -1,     0,     3,    -1,    -1
    };

    vector<float> ref_depth =
    {
           -1.000000,    0.015379,    0.019992,    0.024606,   -1.000000,
            0.007689,    0.014302,   -1.000000,    0.027989,   -1.000000,
            0.005536,    0.013072,    0.021684,    0.030604,    0.032910,
           -1.000000,   -1.000000,    0.018608,    0.029988,    0.037216,
           -1.000000,    0.010765,   -1.000000,    0.023683,   -1.000000
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
TEST(Odometry, ComputeCorrespondence)
{
    vector<Eigen::Vector4i> ref_output =
    {
        {    37,     0,    59,     0 },
        {    98,     4,   120,     4 },
        {   159,     8,   181,     8 },
        {    41,    17,    63,    17 },
        {   102,    21,   124,    21 },
        {   163,    25,   185,    25 },
        {    45,    34,    67,    34 },
        {   106,    38,   128,    38 },
        {   167,    42,   189,    42 },
        {    49,    51,    71,    51 },
        {   110,    55,   132,    55 },
        {   171,    59,   193,    59 },
        {    53,    68,    75,    68 },
        {   114,    72,   136,    72 },
        {   175,    76,   197,    76 },
        {    57,    85,    79,    85 },
        {   118,    89,   140,    89 },
        {   179,    93,   201,    93 },
        {     0,    98,    22,    98 },
        {    61,   102,    83,   102 },
        {   122,   106,   144,   106 },
        {   183,   110,   205,   110 },
        {     4,   115,    26,   115 },
        {    65,   119,    87,   119 },
        {   126,   123,   148,   123 },
        {   187,   127,   209,   127 },
        {     8,   132,    30,   132 },
        {    69,   136,    91,   136 },
        {   130,   140,   152,   140 },
        {   191,   144,   213,   144 },
        {    12,   149,    34,   149 },
        {    73,   153,    95,   153 },
        {   134,   157,   156,   157 },
        {   195,   161,   217,   161 },
        {    16,   166,    38,   166 },
        {    77,   170,    99,   170 },
        {   138,   174,   160,   174 },
        {   199,   178,   221,   178 }
    };

    int width = 240;
    int height = 180;

    Eigen::Matrix3d intrinsic = Eigen::Matrix3d::Zero();
    intrinsic(0, 0) = 0.5;
    intrinsic(1, 1) = 0.65;
    intrinsic(0, 2) = 0.75;
    intrinsic(1, 2) = 0.35;
    intrinsic(2, 2) = 0.9;

    Eigen::Matrix4d extrinsic = Eigen::Matrix4d::Zero();
    extrinsic(0, 0) = 1.0;
    extrinsic(1, 1) = 1.0;
    extrinsic(2, 2) = 1.0;
    extrinsic(0, 3) = 1.0;

    shared_ptr<Image> depth_s = DepthBuffer(width, height, 0.0, 6.0, 0);
    shared_ptr<Image> depth_t = DepthBuffer(width, height, 1.0, 5.0, 0);
    OdometryOption option;
    option.max_depth_diff_ = 0.978100725;

    shared_ptr<vector<Eigen::Vector4i>> output =
        ComputeCorrespondence(intrinsic,
                              extrinsic,
                              *depth_s,
                              *depth_t,
                              option);

    EXPECT_EQ(ref_output.size(), output->size());
    for (size_t i = 0; i < ref_output.size(); i++)
        ExpectEQ(ref_output[i], (*output)[i]);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Odometry, ConvertDepthImageToXYZImage)
{
    vector<float> ref_output =
    {
           -0.029619,   -0.010633,    0.019746,    0.004614,   -0.004969,
            0.009227,    0.045905,   -0.009887,    0.018362,    0.084291,
           -0.010086,    0.018731,    0.139146,   -0.011527,    0.021407,
           -0.006920,    0.004614,    0.004614,    0.003922,    0.007843,
            0.007843,    0.044983,    0.017993,    0.017993,    0.029066,
            0.006459,    0.006459,    0.084567,    0.013010,    0.013010,
           -0.016747,    0.028342,    0.011165,    0.007382,    0.037477,
            0.014764,    0.021453,    0.021783,    0.008581,    0.053979,
            0.030450,    0.011995,    0.145144,    0.056684,    0.022330,
           -0.032249,    0.087651,    0.021499,    0.007474,    0.060942,
            0.014948,    0.041984,    0.068466,    0.016794,    0.014948,
            0.013543,    0.003322,    0.092364,    0.057933,    0.014210,
           -0.000554,    0.002073,    0.000369,    0.002814,    0.031607,
            0.005629,    0.007843,    0.017617,    0.003137,    0.085121,
            0.106220,    0.018916,    0.023391,    0.020208,    0.003599
    };

    int width = 5;
    int height = 5;

    shared_ptr<Image> depth = DepthBuffer(width, height, 0.0, 6.0, 0);

    Eigen::Matrix3d intrinsic = Eigen::Matrix3d::Zero();
    intrinsic(0, 0) = 0.5;
    intrinsic(1, 1) = 0.65;
    intrinsic(0, 2) = 0.75;
    intrinsic(1, 2) = 0.35;
    intrinsic(2, 2) = 0.9;

    shared_ptr<Image> output = ConvertDepthImageToXYZImage(*depth, intrinsic);

    size_t output_size = output->data_.size() / sizeof(float);
    float* const output_data = reinterpret_cast<float*>(&output->data_[0]);
    EXPECT_EQ(ref_output.size(), output_size);
    for (size_t i = 0; i < ref_output.size(); i++)
        EXPECT_NEAR(ref_output[i], output_data[i], THRESHOLD_1E_6);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Odometry, CreateCameraMatrixPyramid)
{
    vector<vector<double>> ref_output =
    {
        {
            0.500000,    0.000000,    0.000000,
            0.000000,    0.650000,    0.000000,
            0.750000,    0.350000,    1.000000,
        },
        {
            0.250000,    0.000000,    0.000000,
            0.000000,    0.325000,    0.000000,
            0.375000,    0.175000,    1.000000,
        },
        {
            0.125000,    0.000000,    0.000000,
            0.000000,    0.162500,    0.000000,
            0.187500,    0.087500,    1.000000,
        },
        {
            0.062500,    0.000000,    0.000000,
            0.000000,    0.081250,    0.000000,
            0.093750,    0.043750,    1.000000,
        },
        {
            0.031250,    0.000000,    0.000000,
            0.000000,    0.040625,    0.000000,
            0.046875,    0.021875,    1.000000,
        }
    };

    open3d::PinholeCameraIntrinsic intrinsic;

    int width = 640;
    int height = 480;

    double fx = 0.5;
    double fy = 0.65;

    double cx = 0.75;
    double cy = 0.35;

    intrinsic.SetIntrinsics(width, height, fx, fy, cx, cy);

    int levels = 5;

    vector<Eigen::Matrix3d> output = CreateCameraMatrixPyramid(intrinsic, levels);

    EXPECT_EQ(ref_output.size(), output.size());
    for (size_t i = 0; i < ref_output.size(); i++)
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                EXPECT_NEAR(ref_output[i][r * 3 + c], output[i](c, r), THRESHOLD_1E_6);
}

// ----------------------------------------------------------------------------
// TODO: fix.
// Test fails validation in TravisCI but works locally.
// The values on the diagonal are less by 14 in TravisCI than when run locally.
// ----------------------------------------------------------------------------
TEST(Odometry, DISABLED_CreateInformationMatrix)
{
    vector<vector<double>> ref_output =
    {
        { 17.769332, -1.677223, -0.014674,  0.000000, -0.013595,  1.548815, },
        { -1.677223, 22.485787, -0.004965,  0.013595,  0.000000, -4.816394, },
        { -0.014674, -0.004965, 23.255035, -1.548815,  4.816394,  0.000000, },
        {  0.000000,  0.013595, -1.548815, 22.000000,  0.000000,  0.000000, },
        { -0.013595,  0.000000,  4.816394,  0.000000, 22.000000,  0.000000, },
        {  1.548815, -4.816394,  0.000000,  0.000000,  0.000000, 22.000000, }
    };

    Eigen::Matrix4d extrinsic = Eigen::Matrix4d::Zero();
    extrinsic(0, 0) = 10.0;
    extrinsic(1, 1) = 10.0;
    extrinsic(2, 2) = 0.2;
    extrinsic(0, 3) = 1.0;

    open3d::PinholeCameraIntrinsic intrinsic;

    int width = 240;
    int height = 180;

    double fx = 0.5;
    double fy = 0.65;

    double cx = 0.75;
    double cy = 0.35;

    intrinsic.SetIntrinsics(width, height, fx, fy, cx, cy);

    shared_ptr<Image> depth_s = DepthBuffer(width, height, 0.0, 6.0, 0);
    shared_ptr<Image> depth_t = DepthBuffer(width, height, 0.0, 4.0, 0);

    OdometryOption option;
    option.max_depth_diff_ = 0.25;

    Eigen::Matrix6d output = CreateInformationMatrix(extrinsic,
                                                     intrinsic,
                                                     *depth_s,
                                                     *depth_t,
                                                     option);

    EXPECT_EQ(6, ref_output.size());
    for (int r = 0; r < 6; r++)
    {
        EXPECT_EQ(6, ref_output[r].size());
        for (int c = 0; c < 6; c++)
            EXPECT_NEAR(ref_output[r][c], output(c, r), THRESHOLD_1E_6);
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Odometry, NormalizeIntensity)
{
    vector<float> ref_image_s_data =
    {
            0.500740,    0.493067,    0.499731,    0.500000,    0.501952,
            0.489701,    0.492057,    0.499462,    0.491047,    0.495827,
            0.494480,    0.497106,    0.492596,    0.495086,    0.502625,
            0.502019,    0.497240,    0.498586,    0.488759,    0.496702,
            0.486605,    0.490442,    0.488624,    0.500135,    0.488961
    };

    vector<float> ref_image_t_data =
    {
            0.500263,    0.499389,    0.500148,    0.500179,    0.500401,
            0.499006,    0.499274,    0.500118,    0.499159,    0.499704,
            0.499550,    0.499849,    0.499335,    0.499619,    0.500478,
            0.500409,    0.499865,    0.500018,    0.498898,    0.499803,
            0.498653,    0.499090,    0.498883,    0.500194,    0.498921
    };

    int width = 5;
    int height = 5;
    int num_of_channels = 1;
    int bytes_per_channel = 4;

    Eigen::Matrix3d intrinsic = Eigen::Matrix3d::Zero();
    intrinsic(0, 0) = 0.5;
    intrinsic(1, 1) = 0.65;
    intrinsic(0, 2) = 0.75;
    intrinsic(1, 2) = 0.35;
    intrinsic(2, 2) = 0.9;

    Eigen::Matrix4d extrinsic = Eigen::Matrix4d::Zero();
    extrinsic(0, 0) = 1.0;
    extrinsic(1, 1) = 1.0;
    extrinsic(2, 2) = 1.0;
    extrinsic(0, 3) = 1.0;

    shared_ptr<Image> depth_s = DepthBuffer(width, height, 0.0, 60.0, 0);
    shared_ptr<Image> depth_t = DepthBuffer(width, height, 1.0, 50.0, 0);
    OdometryOption option;
    option.max_depth_diff_ = 0.978100725;

    shared_ptr<vector<Eigen::Vector4i>> correspondence =
        ComputeCorrespondence(intrinsic,
                              extrinsic,
                              *depth_s,
                              *depth_t,
                              option);

    Image image_s;
    Image image_t;

    image_s.PrepareImage(width, height, num_of_channels, bytes_per_channel);
    image_t.PrepareImage(width, height, num_of_channels, bytes_per_channel);

    float* const image_s_data = reinterpret_cast<float*>(&image_s.data_[0]);
    size_t image_s_size = image_s.data_.size() / sizeof(float);
    Rand(image_s_data, width * height, 10.0, 100.0, 0);
    float* const image_t_data = reinterpret_cast<float*>(&image_t.data_[0]);
    size_t image_t_size = image_t.data_.size() / sizeof(float);
    Rand(image_t_data, width * height, 100.0, 200.0, 0);

    NormalizeIntensity(image_s, image_t, *correspondence);

    EXPECT_EQ(ref_image_s_data.size(), image_s_size);
    for (size_t i = 0; i < ref_image_s_data.size(); i++)
        EXPECT_NEAR(ref_image_s_data[i], image_s_data[i], THRESHOLD_1E_6);

    EXPECT_EQ(ref_image_t_data.size(), image_t_size);
    for (size_t i = 0; i < ref_image_t_data.size(); i++)
        EXPECT_NEAR(ref_image_t_data[i], image_t_data[i], THRESHOLD_1E_6);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Odometry, PackRGBDImage)
{
    int width = 640;
    int height = 480;
    int num_of_channels = 1;
    int bytes_per_channel = 4;

    Image color;

    color.PrepareImage(width,
                       height,
                       num_of_channels,
                       bytes_per_channel);

    Rand(color.data_, 0, 255, 0);

    shared_ptr<Image> depth = DepthBuffer(width, height, 0.0, 60.0, 0);

    shared_ptr<RGBDImage> rgbd_image = PackRGBDImage(color, *depth);

    EXPECT_EQ(rgbd_image->color_.data_.size(), color.data_.size());
    for (size_t i = 0; i < rgbd_image->color_.data_.size(); i++)
        EXPECT_EQ(rgbd_image->color_.data_[i], color.data_[i]);

    EXPECT_EQ(rgbd_image->depth_.data_.size(), depth->data_.size());
    for (size_t i = 0; i < rgbd_image->depth_.data_.size(); i++)
        EXPECT_EQ(rgbd_image->depth_.data_[i], depth->data_[i]);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Odometry, PreprocessDepth)
{
    vector<float> ref_output =
    {
            0.197463,    0.092272,    0.183622,    0.187313,    0.214072,
            0.046136,    0.078431,    0.179931,    0.064591,    0.130104,
            0.111649,    0.147636,    0.085813,    0.119954,    0.223299,
            0.214994,    0.149481,    0.167935,    0.033218,    0.142099,
            0.003691,    0.056286,    0.031373,    0.189158,    0.035986
    };

    int width = 5;
    int height = 5;

    shared_ptr<Image> depth = DepthBuffer(width, height, 0.0, 60.0, 0);

    OdometryOption option;
    option.max_depth_diff_ = 0.978100725;

    shared_ptr<Image> output = PreprocessDepth(*depth, option);

    float* const output_data = reinterpret_cast<float*>(&(*output).data_[0]);
    size_t output_size = output->data_.size() / sizeof(float);

    EXPECT_EQ(ref_output.size(), output_size);
    for (size_t i = 0; i < ref_output.size(); i++)
        EXPECT_NEAR(ref_output[i], output_data[i], THRESHOLD_1E_6);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Odometry, CheckImagePair)
{
    int width = 5;
    int height = 5;

    shared_ptr<Image> depth0 = DepthBuffer(width, height,  0.0, 60.0, 0);
    shared_ptr<Image> depth1 = DepthBuffer(width, height, 10.0, 50.0, 0);

    EXPECT_TRUE(CheckImagePair(*depth0, *depth1));
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Odometry, CheckRGBDImagePair)
{
    int width = 640;
    int height = 480;
    int num_of_channels = 1;
    int bytes_per_channel = 4;

    Image color0;
    Image color1;
    Image depth0;
    Image depth1;

    color0.PrepareImage(width,
                        height,
                        num_of_channels,
                        bytes_per_channel);

    color1.PrepareImage(width,
                        height,
                        num_of_channels,
                        bytes_per_channel);

    depth0.PrepareImage(width,
                        height,
                        num_of_channels,
                        bytes_per_channel);

    depth1.PrepareImage(width,
                        height,
                        num_of_channels,
                        bytes_per_channel);

    shared_ptr<RGBDImage> rgbd_image0 = PackRGBDImage(color0, depth0);
    shared_ptr<RGBDImage> rgbd_image1 = PackRGBDImage(color1, depth1);

    EXPECT_TRUE(CheckRGBDImagePair(*rgbd_image0, *rgbd_image1));
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Odometry, InitializeRGBDOdometry)
{
    vector<float> ref_rgbd0_color =
    {
            0.000000,    0.000000,    0.442811,    0.885622,    0.442811,
            0.499995,    0.999904,    0.647547,    0.295207,    0.147604,
            0.999990,    1.999809,    0.999887,   -0.000000,   -0.000000,
            0.500010,    0.999932,    1.672728,    2.372141,    1.252569,
            0.000044,    0.000084,    3.518353,    7.116423,    3.757708
    };

    vector<float> ref_rgbd1_color =
    {
            0.000000,    0.000000,    0.391780,    0.783561,    0.391780,
            0.442374,    0.884673,    0.572922,    0.261187,    0.130593,
            0.884748,    1.769345,    0.884657,   -0.000000,   -0.000000,
            0.442387,    0.884697,    1.479958,    2.098768,    1.108220,
            0.000039,    0.000074,    3.112888,    6.296306,    3.324659
    };

    vector<float> ref_rgbd0_depth =
    {
            0.141926,    0.129988,    0.152710,    0.172261,    0.183968,
            0.100058,    0.114014,    0.130738,    0.137486,    0.158074,
            0.123529,    0.128143,    0.118743,    0.119608,    0.155882,
            0.133679,    0.125260,    0.111534,    0.109227,    0.125375,
            0.062284,    0.070300,    0.090196,    0.107093,    0.084429
    };

    vector<float> ref_rgbd1_depth =
    {
            0.112284,    0.152710,    0.172261,    0.151326,    0.095156,
            0.113725,    0.130738,    0.137486,    0.144752,    0.134429,
            0.130565,    0.118743,    0.119608,    0.145040,    0.147174,
            0.122607,    0.111534,    0.109227,    0.111188,    0.093310,
            0.076067,    0.090196,    0.107093,    0.086678,    0.069262
    };

    int width = 5;
    int height = 5;
    int num_of_channels = 1;
    int bytes_per_channel = 4;

    Image color0;
    Image color1;

    color0.PrepareImage(width,
                        height,
                        num_of_channels,
                        bytes_per_channel);
    color1.PrepareImage(width,
                        height,
                        num_of_channels,
                        bytes_per_channel);

    Rand(color0.data_, 0, 255, 0);
    Rand(color1.data_, 0, 255, 0);

    shared_ptr<Image> depth0 = DepthBuffer(width, height, 0.0, 60.0, 0);
    shared_ptr<Image> depth1 = DepthBuffer(width, height, 0.0, 60.0, 1);

    shared_ptr<RGBDImage> source = PackRGBDImage(color0, *depth0);
    shared_ptr<RGBDImage> target = PackRGBDImage(color1, *depth1);

    open3d::PinholeCameraIntrinsic intrinsic;

    double fx = 0.5;
    double fy = 0.65;

    double cx = 0.75;
    double cy = 0.35;

    intrinsic.SetIntrinsics(width, height, fx, fy, cx, cy);

    Eigen::Matrix4d extrinsic = Eigen::Matrix4d::Zero();
    extrinsic(0, 0) = 1.0;
    extrinsic(1, 1) = 1.0;
    extrinsic(2, 2) = 1.0;
    extrinsic(0, 3) = 1.0;

    OdometryOption option;
    option.max_depth_diff_ = 0.5;

    shared_ptr<RGBDImage> rgbd0 = NULL;
    shared_ptr<RGBDImage> rgbd1 = NULL;

    tie(rgbd0, rgbd1) = InitializeRGBDOdometry(*source,
                                               *target,
                                               intrinsic,
                                               extrinsic,
                                               option);

    float* const rgbd0_color = reinterpret_cast<float*>(&(*rgbd0).color_.data_[0]);
    float* const rgbd1_color = reinterpret_cast<float*>(&(*rgbd1).color_.data_[0]);
    float* const rgbd0_depth = reinterpret_cast<float*>(&(*rgbd0).depth_.data_[0]);
    float* const rgbd1_depth = reinterpret_cast<float*>(&(*rgbd1).depth_.data_[0]);
    size_t rgbd_size = width * height;

    EXPECT_EQ(ref_rgbd0_color.size(), rgbd_size);
    for (size_t i = 0; i < ref_rgbd0_color.size(); i++)
        EXPECT_NEAR(ref_rgbd0_color[i], rgbd0_color[i], THRESHOLD_1E_6);

    EXPECT_EQ(ref_rgbd1_color.size(), rgbd_size);
    for (size_t i = 0; i < ref_rgbd1_color.size(); i++)
        EXPECT_NEAR(ref_rgbd1_color[i], rgbd1_color[i], THRESHOLD_1E_6);

    EXPECT_EQ(ref_rgbd0_depth.size(), rgbd_size);
    for (size_t i = 0; i < ref_rgbd0_depth.size(); i++)
        EXPECT_NEAR(ref_rgbd0_depth[i], rgbd0_depth[i], THRESHOLD_1E_6);

    EXPECT_EQ(ref_rgbd1_depth.size(), rgbd_size);
    for (size_t i = 0; i < ref_rgbd1_depth.size(); i++)
        EXPECT_NEAR(ref_rgbd1_depth[i], rgbd1_depth[i], THRESHOLD_1E_6);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Odometry, DISABLED_DoSingleIteration)
{
    int iter = 1;
    int level = 1;

    int width = 240;
    int height = 180;
    int num_of_channels = 1;
    int bytes_per_channel = 4;

    shared_ptr<Image> srcColor = GenerateImage(width, height, 1, 4, 0.0f, 1.0f, 1);
    shared_ptr<Image> srcDepth = GenerateImage(width, height, 1, 4, 0.0f, 6.0f, 0);

    shared_ptr<Image> tgtColor = GenerateImage(width, height, 1, 4, 0.0f, 1.0f, 1);
    shared_ptr<Image> tgtDepth = GenerateImage(width, height, 1, 4, 1.0f, 5.0f, 0);

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
    intrinsic(2, 2) = 0.9;

    Eigen::Matrix4d extrinsic = Eigen::Matrix4d::Zero();
    extrinsic(0, 0) = 1.0;
    extrinsic(1, 1) = 1.0;
    extrinsic(2, 2) = 1.0;
    extrinsic(0, 3) = 1.0;

    RGBDOdometryJacobianFromColorTerm jacobian_method;

    OdometryOption option;
    option.max_depth_diff_ = 0.978100725;

    bool status = false;
    Eigen::Matrix4d output = Eigen::Matrix4d::Zero();
    tie(status, output) = DoSingleIteration(iter,
                                            level,
                                            source,
                                            target,
                                            *source_xyz,
                                            target_dx,
                                            target_dy,
                                            intrinsic,
                                            extrinsic,
                                            jacobian_method,
                                            option);

    Print(output);
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
