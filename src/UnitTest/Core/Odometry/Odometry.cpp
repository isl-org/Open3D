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

using namespace Eigen;
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

    int* const map_data = Cast<int>(&map->data_[0]);
    size_t map_size = map->data_.size() / sizeof(int);
    for (int i = 0; i < map_size; i++)
        EXPECT_EQ(-1, map_data[i]);

    float* const depth_data = Cast<float>(&depth->data_[0]);
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
           -1.000000,    3.921569,    5.098040,    6.274510,   -1.000000,
            1.960784,    3.647059,   -1.000000,    7.137255,   -1.000000,
            1.411765,    3.333333,    5.529412,    7.803922,    8.392157,
           -1.000000,   -1.000000,    4.745098,    7.647059,    9.490196,
           -1.000000,    2.745098,   -1.000000,    6.039216,   -1.000000
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

    int* const map_data = Cast<int>(&map->data_[0]);
    ExpectEQ(&ref_map[0], map_data, map->data_.size() / sizeof(int));

    float* const depth_data = Cast<float>(&depth->data_[0]);
    ExpectEQ(&ref_depth[0], depth_data, depth->data_.size() / sizeof(float));
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
            4.035294,    1.352941,    3.682353,    3.776471,    4.458824,
            0.176471,    1.000000,    3.588235,    0.647059,   -1.000000,
           -1.000000,   -1.000000,   -1.000000,   -1.000000,    4.694118,
            4.482353,    2.811765,    3.282353,   -0.152941,    2.623529,
           -0.905882,    0.435294,   -0.200000,    3.823529,   -1.000000
    };

    int width = 5;
    int height = 5;

    shared_ptr<Image> map;
    shared_ptr<Image> depth;

    tie(map, depth) = InitializeCorrespondenceMap(width, height);

    shared_ptr<Image> map_part = CorrespondenceMap(width, height, -1, 5, 0);
    shared_ptr<Image> depth_part = DepthBuffer(width, height, -1.0, 5.0, 0);

    MergeCorrespondenceMaps(*map, *depth, *map_part, *depth_part);

    int* const map_data = Cast<int>(&map->data_[0]);
    ExpectEQ(&ref_map[0], map_data, map->data_.size() / sizeof(int));

    float* const depth_data = Cast<float>(&depth->data_[0]);
    ExpectEQ(&ref_depth[0], depth_data, depth->data_.size() / sizeof(float));
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
    vector<Vector4i> ref_output =
    {
        {    59,     2,    59,     2 },
        {   120,     6,   120,     6 },
        {   181,    10,   181,    10 },
        {     2,    15,     2,    15 },
        {    63,    19,    63,    19 },
        {   124,    23,   124,    23 },
        {   185,    27,   185,    27 },
        {     6,    32,     6,    32 },
        {    67,    36,    67,    36 },
        {   128,    40,   128,    40 },
        {   189,    44,   189,    44 },
        {    10,    49,    10,    49 },
        {    71,    53,    71,    53 },
        {   132,    57,   132,    57 },
        {   193,    61,   193,    61 },
        {    14,    66,    14,    66 },
        {    75,    70,    75,    70 },
        {   136,    74,   136,    74 },
        {   197,    78,   197,    78 },
        {    18,    83,    18,    83 },
        {    79,    87,    79,    87 },
        {   140,    91,   140,    91 },
        {   201,    95,   201,    95 },
        {    22,   100,    22,   100 },
        {    83,   104,    83,   104 },
        {   144,   108,   144,   108 },
        {   205,   112,   205,   112 },
        {    26,   117,    26,   117 },
        {    87,   121,    87,   121 },
        {   148,   125,   148,   125 },
        {   209,   129,   209,   129 },
        {    30,   134,    30,   134 },
        {    91,   138,    91,   138 },
        {   152,   142,   152,   142 },
        {   213,   146,   213,   146 },
        {    34,   151,    34,   151 },
        {    95,   155,    95,   155 },
        {   156,   159,   156,   159 },
        {   217,   163,   217,   163 },
        {    38,   168,    38,   168 },
        {    99,   172,    99,   172 },
        {   160,   176,   160,   176 }
    };

    int width = 240;
    int height = 180;

    Matrix3d intrinsic = Matrix3d::Zero();
    intrinsic(0, 0) = 0.5;
    intrinsic(1, 1) = 0.65;
    intrinsic(0, 2) = 0.75;
    intrinsic(1, 2) = 0.35;
    intrinsic(2, 2) = 0.9;

    Matrix4d extrinsic = Matrix4d::Zero();
    extrinsic(0, 0) = 1.0;
    extrinsic(1, 1) = 1.0;
    extrinsic(2, 2) = 1.0;
    extrinsic(0, 3) = 1.0;

    shared_ptr<Image> depth_s = DepthBuffer(width, height,  0.0, 255.0, 0);
    shared_ptr<Image> depth_t = DepthBuffer(width, height, 10.0, 255.0, 1);
    OdometryOption option;
    option.max_depth_diff_ = 0.25;

    shared_ptr<vector<Vector4i>> output =
        ComputeCorrespondence(intrinsic,
                              extrinsic,
                              *depth_s,
                              *depth_t,
                              option);

    ExpectEQ(ref_output, *output);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Odometry, ConvertDepthImageToXYZImage)
{
    vector<float> ref_output =
    {
           -7.552941,   -2.711312,    5.035294,    1.176471,   -1.266968,
            2.352941,   11.705883,   -2.521267,    4.682353,   21.494118,
           -2.571946,    4.776471,   35.482353,   -2.939367,    5.458824,
           -1.764706,    1.176471,    1.176471,    1.000000,    2.000000,
            2.000000,   11.470589,    4.588235,    4.588235,    7.411765,
            1.647059,    1.647059,   21.564707,    3.317647,    3.317647,
           -4.270588,    7.227149,    2.847059,    1.882353,    9.556561,
            3.764706,    5.470588,    5.554751,    2.188235,   13.764706,
            7.764706,    3.058824,   37.011765,   14.454298,    5.694118,
           -8.223530,   22.351131,    5.482353,    1.905882,   15.540272,
            3.811765,   10.705882,   17.458824,    4.282353,    3.811765,
            3.453394,    0.847059,   23.552940,   14.772851,    3.623529,
           -0.141176,    0.528507,    0.094118,    0.717647,    8.059729,
            1.435294,    2.000000,    4.492308,    0.800000,   21.705881,
           27.085972,    4.823529,    5.964706,    5.152942,    0.917647
    };

    int width = 5;
    int height = 5;

    shared_ptr<Image> depth = DepthBuffer(width, height, 0.0, 6.0, 0);

    Matrix3d intrinsic = Matrix3d::Zero();
    intrinsic(0, 0) = 0.5;
    intrinsic(1, 1) = 0.65;
    intrinsic(0, 2) = 0.75;
    intrinsic(1, 2) = 0.35;
    intrinsic(2, 2) = 0.9;

    shared_ptr<Image> output = ConvertDepthImageToXYZImage(*depth, intrinsic);

    float* const output_data = Cast<float>(&output->data_[0]);
    ExpectEQ(&ref_output[0], output_data, output->data_.size() / sizeof(float));
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Odometry, CreateCameraMatrixPyramid)
{
    vector<Matrix3d> ref_output(5);
    ref_output[0] << 0.500000, 0.000000, 0.750000,
                     0.000000, 0.650000, 0.350000,
                     0.000000, 0.000000, 1.000000;
    ref_output[1] << 0.250000, 0.000000, 0.375000,
                     0.000000, 0.325000, 0.175000,
                     0.000000, 0.000000, 1.000000;
    ref_output[2] << 0.125000, 0.000000, 0.187500,
                     0.000000, 0.162500, 0.087500,
                     0.000000, 0.000000, 1.000000;
    ref_output[3] << 0.062500, 0.000000, 0.093750,
                     0.000000, 0.081250, 0.043750,
                     0.000000, 0.000000, 1.000000;
    ref_output[4] << 0.031250, 0.000000, 0.046875,
                     0.000000, 0.040625, 0.021875,
                     0.000000, 0.000000, 1.000000;

    PinholeCameraIntrinsic intrinsic;

    int width = 640;
    int height = 480;

    double fx = 0.5;
    double fy = 0.65;

    double cx = 0.75;
    double cy = 0.35;

    intrinsic.SetIntrinsics(width, height, fx, fy, cx, cy);

    int levels = 5;

    vector<Matrix3d> output = CreateCameraMatrixPyramid(intrinsic, levels);

    ExpectEQ(ref_output, output);
}

// ----------------------------------------------------------------------------
// TODO: fix.
// Test fails validation in TravisCI but works locally.
// The values on the diagonal are less by 14 in TravisCI than when run locally.
// ----------------------------------------------------------------------------
TEST(Odometry, DISABLED_CreateInformationMatrix)
{
    Matrix6d ref_output;
    ref_output <<
      613.863173, -1502.530770,  -29.912557,   0.000000, -0.486275,  24.425943,
    -1502.530770,  3801.174861,  -11.877714,   0.486275,  0.000000, -61.513725,
      -29.912557,   -11.877714, 4397.565108, -24.425943, 61.513725,   0.000000,
        0.000000,     0.486275,  -24.425943,  18.000000,  0.000000,   0.000000,
       -0.486275,     0.000000,   61.513725,   0.000000, 18.000000,   0.000000,
       24.425943,   -61.513725,    0.000000,   0.000000,  0.000000,  18.000000;

    Matrix4d extrinsic = Matrix4d::Zero();
    extrinsic(0, 0) = 10.0;
    extrinsic(1, 1) = 10.0;
    extrinsic(2, 2) = 0.2;
    extrinsic(0, 3) = 1.0;

    PinholeCameraIntrinsic intrinsic;

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

    Matrix6d output = CreateInformationMatrix(extrinsic,
                                              intrinsic,
                                              *depth_s,
                                              *depth_t,
                                              option);

    ExpectEQ(ref_output, output);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Odometry, NormalizeIntensity)
{
    vector<float> ref_image_s_data =
    {
            1.873711,    0.992268,    1.757732,    1.788660,    2.012887,
            0.605670,    0.876289,    1.726804,    0.760309,    1.309278,
            1.154639,    1.456186,    0.938144,    1.224227,    2.090206,
            2.020619,    1.471649,    1.626289,    0.497423,    1.409794,
            0.250000,    0.690722,    0.481959,    1.804124,    0.520619
    };

    vector<float> ref_image_t_data =
    {
            0.804920,    0.609268,    0.779176,    0.786041,    0.835812,
            0.523455,    0.583524,    0.772311,    0.557780,    0.679634,
            0.645309,    0.712243,    0.597254,    0.660755,    0.852975,
            0.837529,    0.715675,    0.750000,    0.499428,    0.701945,
            0.444508,    0.542334,    0.495995,    0.789474,    0.504577
    };

    int width = 5;
    int height = 5;
    int num_of_channels = 1;
    int bytes_per_channel = 4;

    Matrix3d intrinsic = Matrix3d::Zero();
    intrinsic(0, 0) = 0.5;
    intrinsic(1, 1) = 0.65;
    intrinsic(0, 2) = 0.75;
    intrinsic(1, 2) = 0.35;
    intrinsic(2, 2) = 0.9;

    Matrix4d extrinsic = Matrix4d::Zero();
    extrinsic(0, 0) = 1.0;
    extrinsic(1, 1) = 1.0;
    extrinsic(2, 2) = 1.0;
    extrinsic(0, 3) = 1.0;

    shared_ptr<Image> depth_s = DepthBuffer(width, height, 0.0, 60.0, 0);
    shared_ptr<Image> depth_t = DepthBuffer(width, height, 1.0, 50.0, 0);
    OdometryOption option;
    option.max_depth_diff_ = 0.978100725;

    shared_ptr<vector<Vector4i>> correspondence =
        ComputeCorrespondence(intrinsic,
                              extrinsic,
                              *depth_s,
                              *depth_t,
                              option);

    Image image_s;
    Image image_t;

    image_s.PrepareImage(width, height, num_of_channels, bytes_per_channel);
    image_t.PrepareImage(width, height, num_of_channels, bytes_per_channel);

    float* const image_s_data = Cast<float>(&image_s.data_[0]);
    size_t image_s_size = image_s.data_.size() / sizeof(float);
    Rand(image_s_data, width * height, 10.0, 100.0, 0);
    float* const image_t_data = Cast<float>(&image_t.data_[0]);
    size_t image_t_size = image_t.data_.size() / sizeof(float);
    Rand(image_t_data, width * height, 100.0, 200.0, 0);

    NormalizeIntensity(image_s, image_t, *correspondence);

    EXPECT_EQ(ref_image_s_data.size(), image_s_size);
    ExpectEQ(&ref_image_s_data[0], image_s_data, image_s_size);

    EXPECT_EQ(ref_image_t_data.size(), image_t_size);
    ExpectEQ(&ref_image_t_data[0], image_t_data, image_t_size);
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

    ExpectEQ(rgbd_image->color_.data_, color.data_);
    ExpectEQ(rgbd_image->depth_.data_, depth->data_);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Odometry, PreprocessDepth)
{
    vector<float> ref_output =
    {
          214.00,  100.00,  199.00,  203.00,  232.00,
           50.00,   85.00,  195.00,   70.00,  141.00,
          121.00,  160.00,   93.00,  130.00,  242.00,
          233.00,  162.00,  182.00,   36.00,  154.00,
            4.00,   61.00,   34.00,  205.00,   39.00
    };

    int width = 5;
    int height = 5;

    shared_ptr<Image> depth = DepthBuffer(width, height, 0.0, 255.0, 0);

    OdometryOption option;
    option.min_depth_ = 1.0;
    option.max_depth_ = 250.0;

    shared_ptr<Image> output = PreprocessDepth(*depth, option);

    float* const output_data = Cast<float>(&(*output).data_[0]);
    size_t output_size = output->data_.size() / sizeof(float);

    EXPECT_EQ(ref_output.size(), output_size);
    ExpectEQ(&ref_output[0], output_data, output_size);
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
            0.636246,    0.582730,    0.684592,    0.772234,    0.824716,
            0.448552,    0.511117,    0.586091,    0.616339,    0.708635,
            0.553775,    0.574457,    0.532316,    0.536194,    0.698811,
            0.599276,    0.561531,    0.500000,    0.489659,    0.562048,
            0.279214,    0.315150,    0.404343,    0.480093,    0.378490
    };

    vector<float> ref_rgbd1_color =
    {
            0.513992,    0.699050,    0.788543,    0.692714,    0.435586,
            0.520591,    0.598469,    0.629356,    0.662619,    0.615364,
            0.597677,    0.543559,    0.547518,    0.663939,    0.673706,
            0.561246,    0.510560,    0.500000,    0.508976,    0.427138,
            0.348205,    0.412883,    0.490232,    0.396779,    0.317054
    };

    vector<float> ref_rgbd0_depth =
    {
          153.812500,  140.875000,  165.500000,  186.687500,  199.375000,
          108.437500,  123.562500,  141.687500,  149.000000,  171.312500,
          133.875000,  138.875000,  128.687500,  129.625000,  168.937500,
          144.875000,  135.750000,  120.875000,  118.375000,  135.875000,
           67.500000,   76.187500,   97.750000,  116.062500,   91.500000
    };

    vector<float> ref_rgbd1_depth =
    {
          126.915451,  169.009796,  189.366425,  167.568619,  109.080879,
          128.416672,  146.131134,  153.156860,  160.723038,  149.974258,
          145.950989,  133.640945,  134.541672,  161.023285,  163.245102,
          137.664215,  126.134811,  123.732849,  125.774513,  107.159317,
           89.204659,  103.916672,  121.511032,  100.253677,   82.118874
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

    float* const color0_data = Cast<float>(&color0.data_[0]);
    float* const color1_data = Cast<float>(&color1.data_[0]);
    size_t color_size = width * height;
    Rand(color0_data, color_size, 0.0, 255.0, 0);
    Rand(color1_data, color_size, 0.0, 255.0, 1);

    shared_ptr<Image> depth0 = DepthBuffer(width, height,  0.0, 255.0, 0);
    shared_ptr<Image> depth1 = DepthBuffer(width, height, 10.0, 255.0, 1);

    shared_ptr<RGBDImage> source = PackRGBDImage(color0, *depth0);
    shared_ptr<RGBDImage> target = PackRGBDImage(color1, *depth1);

    PinholeCameraIntrinsic intrinsic;
    intrinsic.intrinsic_matrix_ = Matrix3d::Zero();
    intrinsic.intrinsic_matrix_(0, 0) = 0.5;
    intrinsic.intrinsic_matrix_(1, 1) = 0.65;
    intrinsic.intrinsic_matrix_(0, 2) = 0.75;
    intrinsic.intrinsic_matrix_(1, 2) = 0.35;
    intrinsic.intrinsic_matrix_(2, 2) = 0.9;

    Matrix4d extrinsic = Matrix4d::Zero();
    extrinsic(0, 0) = 1.0;
    extrinsic(1, 1) = 1.0;
    extrinsic(2, 2) = 1.0;
    extrinsic(0, 3) = 1.0;

    OdometryOption option;
    option.min_depth_ = 1.0;
    option.max_depth_ = 250.0;
    option.max_depth_diff_ = 5;

    shared_ptr<RGBDImage> rgbd0 = NULL;
    shared_ptr<RGBDImage> rgbd1 = NULL;

    tie(rgbd0, rgbd1) = InitializeRGBDOdometry(*source,
                                               *target,
                                               intrinsic,
                                               extrinsic,
                                               option);

    float* const rgbd0_color = Cast<float>(&(*rgbd0).color_.data_[0]);
    float* const rgbd1_color = Cast<float>(&(*rgbd1).color_.data_[0]);
    float* const rgbd0_depth = Cast<float>(&(*rgbd0).depth_.data_[0]);
    float* const rgbd1_depth = Cast<float>(&(*rgbd1).depth_.data_[0]);
    size_t rgbd_size = width * height;

    EXPECT_EQ(ref_rgbd0_color.size(), rgbd_size);
    ExpectEQ(&ref_rgbd0_color[0], rgbd0_color, rgbd_size);

    EXPECT_EQ(ref_rgbd1_color.size(), rgbd_size);
    ExpectEQ(&ref_rgbd1_color[0], rgbd1_color, rgbd_size);

    EXPECT_EQ(ref_rgbd0_depth.size(), rgbd_size);
    ExpectEQ(&ref_rgbd0_depth[0], rgbd0_depth, rgbd_size);

    EXPECT_EQ(ref_rgbd1_depth.size(), rgbd_size);
    ExpectEQ(&ref_rgbd1_depth[0], rgbd1_depth, rgbd_size);
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
    ShiftUp(tgtColor, 5);

    ShiftLeft(dxColor, 10);
    ShiftUp(dyColor, 5);

    RGBDImage source(*srcColor, *srcDepth);
    RGBDImage target(*tgtColor, *tgtDepth);
    shared_ptr<Image> source_xyz = GenerateImage(width, height, 3, 4, 0.0f, 1.0f, 0);
    RGBDImage target_dx(*dxColor, *tgtDepth);
    RGBDImage target_dy(*dyColor, *tgtDepth);

    Matrix3d intrinsic = Matrix3d::Zero();
    intrinsic(0, 0) = 0.5;
    intrinsic(1, 1) = 0.65;
    intrinsic(0, 2) = 0.75;
    intrinsic(1, 2) = 0.35;
    intrinsic(2, 2) = 0.9;

    Matrix4d extrinsic = Matrix4d::Zero();
    extrinsic(0, 0) = 1.0;
    extrinsic(1, 1) = 1.0;
    extrinsic(2, 2) = 1.0;
    extrinsic(0, 3) = 1.0;

    RGBDOdometryJacobianFromColorTerm jacobian_method;

    OdometryOption option;
    option.max_depth_diff_ = 0.978100725;

    bool status = false;
    Matrix4d output = Matrix4d::Zero();
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
