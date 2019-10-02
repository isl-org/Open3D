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

#include "Open3D/Camera/PinholeCameraTrajectory.h"
#include "Open3D/Geometry/Image.h"
#include "Open3D/Geometry/RGBDImage.h"
#include "Open3D/Geometry/TriangleMesh.h"
#include "TestUtility/UnitTest.h"

using namespace open3d;
using namespace std;
using namespace unit_test;

/* TODO
As the color_map::ColorMapOptimization subcomponents go back into hiding several
lines of code had to commented out. Do not remove these lines, they may become
useful again after a decision has been made about the way to make these
subcomponents visible to UnitTest.
*/

vector<geometry::Image> GenerateImages(const int& width,
                                       const int& height,
                                       const int& num_of_channels,
                                       const int& bytes_per_channel,
                                       const size_t& size) {
    vector<geometry::Image> images;
    for (size_t i = 0; i < size; i++) {
        geometry::Image image;

        image.Prepare(width, height, num_of_channels, bytes_per_channel);

        if (bytes_per_channel == 4) {
            float* const depthData = reinterpret_cast<float*>(&image.data_[0]);
            Rand(depthData, width * height, 10.0, 100.0, i);
        } else
            Rand(image.data_, 0, 255, i);

        images.push_back(image);
    }

    return images;
}

vector<shared_ptr<geometry::Image>> GenerateSharedImages(
        const int& width,
        const int& height,
        const int& num_of_channels,
        const int& bytes_per_channel,
        const size_t& size) {
    vector<geometry::Image> images = GenerateImages(
            width, height, num_of_channels, bytes_per_channel, size);

    vector<shared_ptr<geometry::Image>> output;
    for (size_t i = 0; i < size; i++)
        output.push_back(make_shared<geometry::Image>(images[i]));

    return output;
}

vector<geometry::RGBDImage> GenerateRGBDImages(const int& width,
                                               const int& height,
                                               const size_t& size) {
    int num_of_channels = 3;
    int bytes_per_channel = 1;
    int depth_num_of_channels = 1;
    int depth_bytes_per_channel = 4;

    vector<geometry::Image> depths =
            GenerateImages(width, height, depth_num_of_channels,
                           depth_bytes_per_channel, size);

    vector<geometry::Image> colors = GenerateImages(
            width, height, num_of_channels, bytes_per_channel, size);

    vector<geometry::RGBDImage> rgbdImages;
    for (size_t i = 0; i < size; i++) {
        geometry::RGBDImage rgbdImage(colors[i], depths[i]);
        rgbdImages.push_back(rgbdImage);
    }

    return rgbdImages;
}

camera::PinholeCameraTrajectory GenerateCamera(const int& width,
                                               const int& height,
                                               const Eigen::Vector3d& pose) {
    camera::PinholeCameraTrajectory camera;
    camera.parameters_.resize(1);

    double fx = 0.5;
    double fy = 0.65;

    double cx = 0.75;
    double cy = 0.35;

    camera.parameters_[0].intrinsic_.SetIntrinsics(width, height, fx, fy, cx,
                                                   cy);

    camera.parameters_[0].extrinsic_ << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

    camera.parameters_[0].extrinsic_(0, 0) = pose(0, 0);
    camera.parameters_[0].extrinsic_(1, 1) = pose(1, 0);
    camera.parameters_[0].extrinsic_(2, 2) = pose(2, 0);

    return camera;
}

TEST(ColorMapOptimization, DISABLED_Project3DPointAndGetUVDepth) {
    vector<Eigen::Vector3d> ref_points = {
            {1.072613, 0.611307, 42.921570}, {0.897783, 0.859754, 43.784313},
            {1.452353, 1.769294, 18.333334}, {1.181915, 0.663475, 30.411764},
            {1.498387, 0.741398, 20.058823}, {0.814378, 0.620043, 50.254902},
            {1.458333, 1.693333, 7.764706},  {1.709016, 2.412951, 13.156863},
            {1.288462, 2.510000, 8.411765},  {2.316667, 1.043333, 5.823529},
            {1.029231, 0.366000, 28.039215}, {1.390000, 0.585733, 16.176470},
            {0.973200, 0.512240, 26.960785}, {0.948980, 0.437551, 42.274509},
            {1.461765, 1.644902, 22.000000}, {1.535393, 1.109551, 19.196079},
            {3.608824, 5.121765, 3.666667},  {3.350000, 4.361429, 4.529412},
            {0.797577, 0.636344, 48.960785}, {9.990000, 8.046000, 1.078431},
            {0.770000, 1.511333, 12.941176}, {0.834722, 0.595556, 46.588234},
            {0.857368, 0.744105, 20.490196}, {1.111765, 0.977059, 36.666668},
            {0.855405, 0.429640, 23.941177}, {0.917213, 0.730765, 39.470589},
            {0.810736, 0.506319, 35.156864}, {0.942857, 3.160476, 9.058824},
            {1.111137, 0.389431, 45.509804}, {0.822687, 0.615727, 48.960785}};

    int width = 320;
    int height = 240;

    // Eigen::Vector3d point = {3.3, 4.4, 5.5};

    vector<Eigen::Vector3d> output;
    for (size_t i = 0; i < ref_points.size(); i++) {
        // change the pose randomly
        Eigen::Vector3d pose;
        Rand(pose, 0.0, 10.0, i);
        camera::PinholeCameraTrajectory camera =
                GenerateCamera(width, height, pose);

        // float u, v, d;
        // tie(u, v, d) = Project3DPointAndGetUVDepth(point, camera, camid);
        // ExpectEQ(ref_points[i], Eigen::Vector3d(u, v, d));
    }
}

TEST(ColorMapOptimization, DISABLED_MakeVertexAndImageVisibility) {
    vector<vector<int>> ref_second = {
            {1,   421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432,
             433, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471,
             472, 473, 474, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508,
             509, 510, 511, 512, 513, 514, 515, 527, 528, 536, 537, 538, 539,
             540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552,
             553, 554, 555, 556, 557, 562, 563, 564, 565, 566, 567, 568, 569,
             570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582,
             583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595,
             596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608,
             609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621,
             622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634,
             635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647,
             648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660,
             661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673,
             674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686,
             687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699,
             700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712,
             713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725,
             726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738,
             739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751,
             752, 753, 754, 755, 756, 757, 758, 759, 760, 761}};

    int width = 320;
    int height = 240;
    int num_of_channels = 3;
    int bytes_per_channel = 1;
    size_t size = 10;

    shared_ptr<geometry::TriangleMesh> mesh =
            geometry::TriangleMesh::CreateSphere(10.0, 20);
    vector<geometry::RGBDImage> images_rgbd =
            GenerateRGBDImages(width, height, size);
    vector<geometry::Image> images_mask = GenerateImages(
            width, height, num_of_channels, bytes_per_channel, size);

    Eigen::Vector3d pose(-30, -15, -13);
    camera::PinholeCameraTrajectory camera =
            GenerateCamera(width, height, pose);

    // ColorMapOptimizationOption option(false, 4, 0.316, 30, 15, 120, 0.1, 3);

    vector<vector<int>> first;
    vector<vector<int>> second;
    // tie(first, second) = MakeVertexAndImageVisibility(*mesh,
    //                                                   images_rgbd,
    //                                                   images_mask,
    //                                                   camera,
    //                                                   option);

    // first is a large vector of (mostly) empty vectors.
    // TODO: perhaps a different kind of initialization is necessary in order
    // to fill the first vector with data that can be used for validation
    EXPECT_EQ(762, int(first.size()));

    EXPECT_EQ(ref_second.size(), second.size());
    EXPECT_EQ(ref_second[0].size(), second[0].size());
    for (size_t i = 0; i < ref_second[0].size(); i++)
        EXPECT_EQ(ref_second[0][i], second[0][i]);
}

TEST(ColorMapOptimization, DISABLED_MakeWarpingFields) {
    // int ref_anchor_w = 4;
    // int ref_anchor_h = 4;
    // double ref_anchor_step = 1.666667;
    // vector<double> ref_flow = {0.000000, 0.000000, 1.666667,
    // 0.000000, 3.333333,
    //                            0.000000, 5.000000, 0.000000,
    //                            0.000000, 1.666667, 1.666667,
    //                            1.666667, 3.333333, 1.666667, 5.000000,
    //                            1.666667, 0.000000, 3.333333,
    //                            1.666667, 3.333333, 3.333333,
    //                            3.333333, 5.000000, 3.333333, 0.000000,
    //                            5.000000, 1.666667, 5.000000,
    //                            3.333333, 5.000000, 5.000000, 5.000000};

    size_t size = 10;
    int width = 5;
    int height = 5;
    int num_of_channels = 3;
    int bytes_per_channel = 1;

    vector<shared_ptr<geometry::Image>> images = GenerateSharedImages(
            width, height, num_of_channels, bytes_per_channel, size);

    // ColorMapOptimizationOption option(false, 4, 0.316, 30, 2.5, 0.03, 0.1,
    // 3);

    // vector<ImageWarpingField> fields = MakeWarpingFields(images, option);

    // for (size_t i = 0; i < fields.size(); i++)
    // {
    //     EXPECT_EQ(ref_anchor_w, fields[i].anchor_w_);
    //     EXPECT_EQ(ref_anchor_h, fields[i].anchor_h_);
    //     EXPECT_NEAR(ref_anchor_step, fields[i].anchor_step_, THRESHOLD_1E_6);

    //     EXPECT_EQ(ref_flow.size(), fields[i].flow_.size());
    //     for (size_t j = 0; j < fields[i].flow_.size(); j++)
    //         EXPECT_NEAR(ref_flow[j], fields[i].flow_[j], THRESHOLD_1E_6);
    // }
}

TEST(ColorMapOptimization, DISABLED_QueryImageIntensity) {
    vector<bool> ref_bool = {0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0,
                             0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0};

    vector<Eigen::Vector3d> ref_float = {{0.000000, 0.000000, 0.000000},
                                         {10.260207, 10.070588, 10.323875},
                                         {10.257440, 10.114879, 10.260207},
                                         {0.000000, 0.000000, 0.000000},
                                         {10.285121, 10.244983, 10.109343},
                                         {0.000000, 0.000000, 0.000000},
                                         {10.229758, 10.322492, 10.073357},
                                         {10.300346, 10.211764, 10.112111},
                                         {10.229758, 10.113495, 10.211764},
                                         {10.261592, 10.318339, 10.346021},
                                         {0.000000, 0.000000, 0.000000},
                                         {0.000000, 0.000000, 0.000000},
                                         {0.000000, 0.000000, 0.000000},
                                         {0.000000, 0.000000, 0.000000},
                                         {10.157785, 10.347404, 10.249135},
                                         {10.343252, 10.102422, 10.271280},
                                         {10.094118, 10.066436, 10.243599},
                                         {10.024914, 10.001384, 10.325259},
                                         {0.000000, 0.000000, 0.000000},
                                         {10.073357, 10.159169, 10.055364},
                                         {0.000000, 0.000000, 0.000000},
                                         {0.000000, 0.000000, 0.000000},
                                         {0.000000, 0.000000, 0.000000},
                                         {10.217301, 10.098269, 10.276816},
                                         {0.000000, 0.000000, 0.000000}};

    int width = 320;
    int height = 240;
    int num_of_channels = 3;
    int bytes_per_channel = 4;

    geometry::Image img;
    img.Prepare(width, height, num_of_channels, bytes_per_channel);
    float* const depthData = reinterpret_cast<float*>(&img.data_[0]);
    Rand(depthData, width * height, 10.0, 100.0, 0);

    Eigen::Vector3d pose(62.5, 37.5, 1.85);
    camera::PinholeCameraTrajectory camera =
            GenerateCamera(width, height, pose);
    // int camid = 0;
    // int ch = -1;

    size_t size = 25;

    for (size_t i = 0; i < size; i++) {
        vector<double> vData(3);
        Rand(vData, 10.0, 100.0, i);
        Eigen::Vector3d V(vData[0], vData[1], vData[2]);

        // bool boolResult = false;
        // float floatResult = 0.0;

        // tie(boolResult, floatResult) = QueryImageIntensity<float>(img,
        //                                                           V,
        //                                                           camera,
        //                                                           camid,
        //                                                           0);
        // EXPECT_EQ(ref_bool[i], boolResult);
        // EXPECT_NEAR(ref_float[i](0, 0), floatResult, THRESHOLD_1E_6);

        // tie(boolResult, floatResult) = QueryImageIntensity<float>(img,
        //                                                           V,
        //                                                           camera,
        //                                                           camid,
        //                                                           1);
        // EXPECT_EQ(ref_bool[i], boolResult);
        // EXPECT_NEAR(ref_float[i](1, 0), floatResult, THRESHOLD_1E_6);

        // tie(boolResult, floatResult) = QueryImageIntensity<float>(img,
        //                                                           V,
        //                                                           camera,
        //                                                           camid,
        //                                                           2);
        // EXPECT_EQ(ref_bool[i], boolResult);
        // EXPECT_NEAR(ref_float[i](2, 0), floatResult, THRESHOLD_1E_6);
    }
}

TEST(ColorMapOptimization, DISABLED_QueryImageIntensity_WarpingField) {
    vector<bool> ref_bool = {0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0,
                             0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0};

    vector<Eigen::Vector3d> ref_float = {{0.000000, 0.000000, 0.000000},
                                         {10.260207, 10.070588, 10.323875},
                                         {10.257440, 10.114879, 10.260207},
                                         {0.000000, 0.000000, 0.000000},
                                         {10.285121, 10.244983, 10.109343},
                                         {0.000000, 0.000000, 0.000000},
                                         {10.229758, 10.322492, 10.073357},
                                         {10.300346, 10.211764, 10.112111},
                                         {10.229758, 10.113495, 10.211764},
                                         {10.261592, 10.318339, 10.346021},
                                         {0.000000, 0.000000, 0.000000},
                                         {0.000000, 0.000000, 0.000000},
                                         {0.000000, 0.000000, 0.000000},
                                         {0.000000, 0.000000, 0.000000},
                                         {10.157785, 10.347404, 10.249135},
                                         {10.343252, 10.102422, 10.271280},
                                         {10.094118, 10.066436, 10.243599},
                                         {10.024914, 10.001384, 10.325259},
                                         {0.000000, 0.000000, 0.000000},
                                         {10.073357, 10.159169, 10.055364},
                                         {0.000000, 0.000000, 0.000000},
                                         {0.000000, 0.000000, 0.000000},
                                         {0.000000, 0.000000, 0.000000},
                                         {10.217301, 10.098269, 10.276816},
                                         {0.000000, 0.000000, 0.000000}};

    int width = 320;
    int height = 240;
    int num_of_channels = 3;
    int bytes_per_channel = 4;

    geometry::Image img;
    img.Prepare(width, height, num_of_channels, bytes_per_channel);
    float* const depthData = reinterpret_cast<float*>(&img.data_[0]);
    Rand(depthData, width * height, 10.0, 100.0, 0);

    // TODO: change the initialization in such a way that the field has an
    // effect on the outcome of QueryImageIntensity.
    // int nr_anchors = 16;
    // open3d::ImageWarpingField field(width, height, nr_anchors);

    Eigen::Vector3d pose(62.5, 37.5, 1.85);
    camera::PinholeCameraTrajectory camera =
            GenerateCamera(width, height, pose);
    // int camid = 0;
    // int ch = -1;

    size_t size = 25;

    for (size_t i = 0; i < size; i++) {
        vector<double> vData(3);
        Rand(vData, 10.0, 100.0, i);
        Eigen::Vector3d V(vData[0], vData[1], vData[2]);

        // bool boolResult = false;
        // float floatResult = 0.0;

        // tie(boolResult, floatResult) = QueryImageIntensity<float>(img,
        //                                                           V,
        //                                                           camera,
        //                                                           camid,
        //                                                           0);
        // EXPECT_EQ(ref_bool[i], boolResult);
        // EXPECT_NEAR(ref_float[i](0, 0), floatResult, THRESHOLD_1E_6);

        // tie(boolResult, floatResult) = QueryImageIntensity<float>(img,
        //                                                           V,
        //                                                           camera,
        //                                                           camid,
        //                                                           1);
        // EXPECT_EQ(ref_bool[i], boolResult);
        // EXPECT_NEAR(ref_float[i](1, 0), floatResult, THRESHOLD_1E_6);

        // tie(boolResult, floatResult) = QueryImageIntensity<float>(img,
        //                                                           V,
        //                                                           camera,
        //                                                           camid,
        //                                                           2);
        // EXPECT_EQ(ref_bool[i], boolResult);
        // EXPECT_NEAR(ref_float[i](2, 0), floatResult, THRESHOLD_1E_6);
    }
}

TEST(ColorMapOptimization, DISABLED_SetProxyIntensityForVertex) {
    vector<double> ref_proxy_intensity = {
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            10.120416, 10.113495, 10.192388, 0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  10.244983, 10.272664, 10.304499, 10.328028, 0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  10.262976, 10.120416, 10.106574,
            10.181314, 0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  10.262976,
            10.120416, 10.106574, 10.181314, 0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  10.244983, 10.272664, 10.304499, 10.328028, 0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  10.120416, 10.113495,
            10.192388, 0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000};

    size_t size = 10;

    int width = 320;
    int height = 240;
    int num_of_channels = 1;
    int bytes_per_channel = 4;

    shared_ptr<geometry::TriangleMesh> mesh =
            geometry::TriangleMesh::CreateSphere(10.0, 10);

    vector<shared_ptr<geometry::Image>> images_gray = GenerateSharedImages(
            width, height, num_of_channels, bytes_per_channel, size);

    Eigen::Vector3d pose(30, 15, 0.3);
    camera::PinholeCameraTrajectory camera =
            GenerateCamera(width, height, pose);
    // int camid = 0;

    int n_vertex = mesh->vertices_.size();
    vector<vector<int>> vertex_to_image(n_vertex, vector<int>(size, 0));

    vector<double> proxy_intensity;

    // SetProxyIntensityForVertex(*mesh,
    //                            images_gray,
    //                            camera,
    //                            vertex_to_image,
    //                            proxy_intensity);

    EXPECT_EQ(ref_proxy_intensity.size(), proxy_intensity.size());
    for (size_t i = 0; i < proxy_intensity.size(); i++)
        EXPECT_NEAR(ref_proxy_intensity[i], proxy_intensity[i], THRESHOLD_1E_6);
}

TEST(ColorMapOptimization, DISABLED_SetProxyIntensityForVertex_WarpingField) {
    vector<double> ref_proxy_intensity = {
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            10.120416, 10.113495, 10.192388, 0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  10.244983, 10.272664, 10.304499, 10.328028, 0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  10.262976, 10.120416, 10.106574,
            10.181314, 0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  10.262976,
            10.120416, 10.106574, 10.181314, 0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  10.244983, 10.272664, 10.304499, 10.328028, 0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  10.120416, 10.113495,
            10.192388, 0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000};

    size_t size = 10;

    int width = 320;
    int height = 240;
    int num_of_channels = 1;
    int bytes_per_channel = 4;

    shared_ptr<geometry::TriangleMesh> mesh =
            geometry::TriangleMesh::CreateSphere(10.0, 10);

    // TODO: change the initialization in such a way that the fields have an
    // effect on the outcome of QueryImageIntensity.
    // int nr_anchors = 6;
    // vector<ImageWarpingField> fields;
    // for (size_t i = 0; i < size; i++)
    // {
    //     ImageWarpingField field(width, height, nr_anchors + i);
    //     fields.push_back(field);
    // }

    vector<shared_ptr<geometry::Image>> images_gray = GenerateSharedImages(
            width, height, num_of_channels, bytes_per_channel, size);

    Eigen::Vector3d pose(30, 15, 0.3);
    camera::PinholeCameraTrajectory camera =
            GenerateCamera(width, height, pose);
    // int camid = 0;

    int n_vertex = mesh->vertices_.size();
    vector<vector<int>> vertex_to_image(n_vertex, vector<int>(size, 0));

    vector<double> proxy_intensity;

    // SetProxyIntensityForVertex(*mesh,
    //                            images_gray,
    //                            fields,
    //                            camera,
    //                            vertex_to_image,
    //                            proxy_intensity);

    EXPECT_EQ(ref_proxy_intensity.size(), proxy_intensity.size());
    for (size_t i = 0; i < proxy_intensity.size(); i++)
        EXPECT_NEAR(ref_proxy_intensity[i], proxy_intensity[i], THRESHOLD_1E_6);
}
TEST(ColorMapOptimization, DISABLED_OptimizeImageCoorNonrigid) {
    vector<double> ref_proxy_intensity = {
            0.000000,  0.000000,  0.000000,  10.319723, 10.134256, 0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            10.206228, 10.059516, 10.102422, 0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  0.000000,  10.206228, 10.059516, 10.102422,
            0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,  10.319723, 10.134256, 0.000000,  0.000000};

    size_t size = 1;
    int width = 320;
    int height = 240;
    int num_of_channels = 1;
    int bytes_per_channel = 4;

    shared_ptr<geometry::TriangleMesh> mesh =
            geometry::TriangleMesh::CreateSphere(10.0, 5);

    vector<shared_ptr<geometry::Image>> images_gray = GenerateSharedImages(
            width, height, num_of_channels, bytes_per_channel, size);

    vector<shared_ptr<geometry::Image>> images_dx = GenerateSharedImages(
            width, height, num_of_channels, bytes_per_channel, size);

    vector<shared_ptr<geometry::Image>> images_dy = GenerateSharedImages(
            width, height, num_of_channels, bytes_per_channel, size);
    // int nr_anchors = 6;
    // vector<ImageWarpingField> warping_fields;
    // for (size_t i = 0; i < size; i++)
    // {
    //     ImageWarpingField field(width, height, nr_anchors + i);
    //     warping_fields.push_back(field);
    // }
    // vector<ImageWarpingField> warping_fields_init;
    // for (size_t i = 0; i < size; i++)
    // {
    //     ImageWarpingField field(width, height, nr_anchors + i);
    //     warping_fields_init.push_back(field);
    // }

    Eigen::Vector3d pose(60, 15, 0.3);
    camera::PinholeCameraTrajectory camera =
            GenerateCamera(width, height, pose);

    size_t n_vertex = mesh->vertices_.size();
    size_t n_camera = camera.parameters_.size();

    vector<vector<int>> vertex_to_image(n_vertex, vector<int>(n_camera, 0));
    vector<vector<int>> image_to_vertex(n_camera, vector<int>(n_vertex, 0));

    for (size_t i = 0; i < image_to_vertex.size(); i++)
        Rand(image_to_vertex[i], 0, n_vertex, i);

    // ColorMapOptimizationOption option(false, 62, 0.316, 30, 2.5, 0.03, 0.95,
    // 3);

    vector<double> proxy_intensity;

    // OptimizeImageCoorNonrigid(
    //     *mesh,
    //     images_gray,
    //     images_dx,
    //     images_dy,
    //     warping_fields,
    //     warping_fields_init,
    //     camera,
    //     vertex_to_image,
    //     image_to_vertex,
    //     proxy_intensity,
    //     option);

    EXPECT_EQ(ref_proxy_intensity.size(), proxy_intensity.size());
    for (size_t i = 0; i < proxy_intensity.size(); i++)
        EXPECT_NEAR(ref_proxy_intensity[i], proxy_intensity[i], THRESHOLD_1E_6);
}

TEST(ColorMapOptimization, DISABLED_OptimizeImageCoorRigid) {
    vector<double> ref_proxy_intensity = {
            0.000000, 0.000000,  0.000000,  10.120416, 10.192388, 0.000000,
            0.000000, 0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000, 10.120416, 10.056747, 0.000000,  0.000000,  0.000000,
            0.000000, 0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000, 0.000000,  0.000000,  0.000000,  10.120416, 10.056747,
            0.000000, 0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
            0.000000, 0.000000,  10.120416, 10.192388, 0.000000,  0.000000};

    size_t size = 1;
    int width = 320;
    int height = 240;
    int num_of_channels = 1;
    int bytes_per_channel = 4;

    shared_ptr<geometry::TriangleMesh> mesh =
            geometry::TriangleMesh::CreateSphere(10.0, 5);

    vector<shared_ptr<geometry::Image>> images_gray = GenerateSharedImages(
            width, height, num_of_channels, bytes_per_channel, size);

    vector<shared_ptr<geometry::Image>> images_dx = GenerateSharedImages(
            width, height, num_of_channels, bytes_per_channel, size);

    vector<shared_ptr<geometry::Image>> images_dy = GenerateSharedImages(
            width, height, num_of_channels, bytes_per_channel, size);

    Eigen::Vector3d pose(30, 15, 0.3);
    camera::PinholeCameraTrajectory camera =
            GenerateCamera(width, height, pose);

    size_t n_vertex = mesh->vertices_.size();
    size_t n_camera = camera.parameters_.size();

    vector<vector<int>> vertex_to_image(n_vertex, vector<int>(n_camera, 0));
    vector<vector<int>> image_to_vertex(n_camera, vector<int>(n_vertex, 0));

    for (size_t i = 0; i < image_to_vertex.size(); i++)
        Rand(image_to_vertex[i], 0, n_vertex, i);

    // ColorMapOptimizationOption option(false, 62, 0.316, 30, 2.5, 0.03, 0.95,
    // 3);

    vector<double> proxy_intensity;

    // OptimizeImageCoorRigid(
    //     *mesh,
    //     images_gray,
    //     images_dx,
    //     images_dy,
    //     camera,
    //     vertex_to_image,
    //     image_to_vertex,
    //     proxy_intensity,
    //     option);

    EXPECT_EQ(ref_proxy_intensity.size(), proxy_intensity.size());
    for (size_t i = 0; i < proxy_intensity.size(); i++)
        EXPECT_NEAR(ref_proxy_intensity[i], proxy_intensity[i], THRESHOLD_1E_6);
}

TEST(ColorMapOptimization, DISABLED_SetGeometryColorAverage) {
    vector<Eigen::Vector3d> ref_vertex_colors = {
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.325490, 0.737255, 0.200000},
            {0.290196, 0.243137, 0.909804}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.925490, 0.105882, 0.384314},
            {0.674510, 0.149020, 0.031373}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.925490, 0.105882, 0.384314}, {0.674510, 0.149020, 0.031373},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.325490, 0.737255, 0.200000}, {0.290196, 0.243137, 0.909804},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000}};

    size_t size = 10;
    int width = 320;
    int height = 240;

    shared_ptr<geometry::TriangleMesh> mesh =
            geometry::TriangleMesh::CreateSphere(10.0, 5);

    vector<geometry::RGBDImage> images_rgbd =
            GenerateRGBDImages(width, height, size);

    Eigen::Vector3d pose(30, 15, 0.3);
    camera::PinholeCameraTrajectory camera =
            GenerateCamera(width, height, pose);
    // int camid = 0;

    int n_vertex = mesh->vertices_.size();
    vector<vector<int>> vertex_to_image(n_vertex, vector<int>(size, 0));

    vector<double> proxy_intensity;

    // SetGeometryColorAverage(*mesh,
    //                         images_rgbd,
    //                         camera,
    //                         vertex_to_image);

    EXPECT_EQ(ref_vertex_colors.size(), mesh->vertex_colors_.size());
    for (size_t i = 0; i < mesh->vertex_colors_.size(); i++)
        ExpectEQ(ref_vertex_colors[i], mesh->vertex_colors_[i]);
}

TEST(ColorMapOptimization, DISABLED_SetGeometryColorAverage_WarpingFields) {
    vector<Eigen::Vector3d> ref_vertex_colors = {
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.325490, 0.737255, 0.200000},
            {0.290196, 0.243137, 0.909804}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.925490, 0.105882, 0.384314},
            {0.674510, 0.149020, 0.031373}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.925490, 0.105882, 0.384314}, {0.674510, 0.149020, 0.031373},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.325490, 0.737255, 0.200000}, {0.290196, 0.243137, 0.909804},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000}};

    size_t size = 10;
    int width = 320;
    int height = 240;

    shared_ptr<geometry::TriangleMesh> mesh =
            geometry::TriangleMesh::CreateSphere(10.0, 5);

    // TODO: change the initialization in such a way that the fields have an
    // effect on the outcome of QueryImageIntensity.
    // int nr_anchors = 6;
    // vector<ImageWarpingField> fields;
    // for (size_t i = 0; i < size; i++)
    // {
    //     ImageWarpingField field(width, height, nr_anchors + i);
    //     fields.push_back(field);
    // }

    vector<geometry::RGBDImage> images_rgbd =
            GenerateRGBDImages(width, height, size);

    Eigen::Vector3d pose(30, 15, 0.3);
    camera::PinholeCameraTrajectory camera =
            GenerateCamera(width, height, pose);
    // int camid = 0;

    int n_vertex = mesh->vertices_.size();
    vector<vector<int>> vertex_to_image(n_vertex, vector<int>(size, 0));

    vector<double> proxy_intensity;

    // SetGeometryColorAverage(*mesh,
    //                         images_rgbd,
    //                         fields,
    //                         camera,
    //                         vertex_to_image);

    EXPECT_EQ(ref_vertex_colors.size(), mesh->vertex_colors_.size());
    for (size_t i = 0; i < mesh->vertex_colors_.size(); i++)
        ExpectEQ(ref_vertex_colors[i], mesh->vertex_colors_[i]);
}

TEST(ColorMapOptimization, DISABLED_MakeGradientImages) {
    vector<vector<double>> ref_images0_data = {
            {76,  133, 33,  63,  159, 183, 31,  63,  11,  93,  9,   63,  164,
             86,  251, 62,  126, 154, 6,   63,  195, 48,  37,  63,  159, 99,
             12,  63,  102, 11,  239, 62,  152, 253, 247, 62,  111, 102, 18,
             63,  142, 55,  41,  63,  8,   14,  15,  63,  2,   41,  0,   63,
             128, 86,  4,   63,  174, 79,  13,  63,  186, 95,  43,  63,  225,
             28,  18,  63,  61,  241, 253, 62,  88,  181, 255, 62,  160, 63,
             2,   63,  152, 47,  44,  63,  189, 191, 16,  63,  51,  190, 2,
             63,  90,  127, 11,  63,  245, 80,  30,  63},
            {29,  204, 21,  63,  234, 221, 239, 62,  190, 46,  228, 62,  1,
             152, 8,   63,  95,  35,  42,  63,  42,  99,  17,  63,  40,  28,
             238, 62,  103, 222, 228, 62,  224, 172, 241, 62,  57,  245, 5,
             63,  176, 48,  32,  63,  104, 118, 11,  63,  104, 192, 253, 62,
             228, 179, 227, 62,  23,  27,  209, 62,  23,  94,  57,  63,  24,
             44,  28,  63,  154, 79,  251, 62,  102, 143, 203, 62,  221, 241,
             207, 62,  198, 46,  72,  63,  30,  83,  27,  63,  208, 111, 238,
             62,  31,  6,   222, 62,  221, 77,  25,  63}};

    vector<vector<double>> ref_images1_data = {
            {88,  177, 241, 189, 166, 71,  236, 190, 116, 19,  237, 190, 168,
             200, 148, 61,  156, 212, 68,  62,  246, 34,  155, 190, 237, 140,
             28,  191, 246, 26,  155, 190, 46,  75,  128, 62,  128, 222, 122,
             62,  18,  198, 204, 190, 68,  47,  44,  191, 96,  88,  96,  190,
             86,  212, 97,  62,  204, 251, 42,  62,  38,  62,  208, 190, 40,
             78,  43,  191, 84,  241, 81,  190, 196, 29,  61,  62,  196, 82,
             2,   62,  212, 36,  215, 190, 74,  187, 40,  191, 120, 13,  8,
             190, 143, 254, 171, 62,  20,  103, 107, 62},
            {28,  217, 231, 190, 48,  18,  10,  191, 0,   14,  79,  62,  6,
             170, 59,  63,  198, 129, 227, 62,  56,  131, 206, 190, 39,  237,
             2,   191, 64,  86,  172, 188, 197, 138, 145, 62,  38,  242, 73,
             62,  74,  247, 193, 190, 56,  75,  29,  191, 234, 169, 207, 190,
             168, 56,  59,  190, 32,  50,  82,  188, 220, 243, 247, 190, 238,
             179, 105, 191, 78,  181, 50,  191, 132, 211, 252, 189, 120, 131,
             21,  62,  247, 196, 35,  191, 114, 77,  151, 191, 144, 84,  59,
             191, 1,   38,  161, 62,  164, 17,  1,   63}};

    vector<vector<double>> ref_images2_data = {
            {192, 25,  5,   189, 128, 79,  83,  190, 224, 188, 98,  190, 64,
             71,  22,  189, 48,  221, 6,   62,  128, 165, 205, 60,  208, 83,
             11,  190, 224, 51,  227, 189, 64,  120, 45,  61,  224, 85,  214,
             61,  64,  49,  194, 61,  64,  163, 200, 61,  192, 215, 195, 61,
             0,   32,  124, 187, 48,  98,  50,  190, 64,  157, 41,  61,  64,
             10,  15,  61,  0,   79,  96,  61,  176, 160, 7,   62,  192, 178,
             104, 62,  0,   58,  137, 59,  0,   148, 237, 59,  160, 149, 142,
             61,  16,  129, 92,  62,  88,  177, 191, 62},
            {192, 187, 97,  189, 0,   92,  186, 188, 128, 44,  127, 189, 232,
             178, 134, 190, 8,   152, 248, 190, 176, 212, 74,  62,  56,  241,
             120, 62,  112, 216, 178, 61,  56,  146, 196, 190, 140, 127, 91,
             191, 204, 14,  29,  63,  28,  223, 250, 62,  224, 1,   34,  62,
             168, 132, 99,  190, 60,  7,   218, 190, 248, 214, 7,   63,  104,
             30,  128, 62,  0,   177, 18,  188, 0,   169, 13,  62,  20,  106,
             15,  63,  80,  100, 46,  62,  128, 218, 213, 60,  192, 172, 143,
             188, 8,   111, 117, 62,  42,  58,  29,  63}};

    size_t size = 2;

    int width = 5;
    int height = 5;

    vector<geometry::RGBDImage> images_rgbd =
            GenerateRGBDImages(width, height, size);

    vector<shared_ptr<geometry::Image>> images0;
    vector<shared_ptr<geometry::Image>> images1;
    vector<shared_ptr<geometry::Image>> images2;

    // tie(images0, images1, images2) = MakeGradientImages(images_rgbd);

    EXPECT_EQ(size, images0.size());
    for (size_t i = 0; i < size; i++) {
        EXPECT_EQ(100u, images0[i]->data_.size());
        for (size_t j = 0; j < images0[i]->data_.size(); j++)
            EXPECT_EQ(ref_images0_data[i][j], images0[i]->data_[j]);
    }

    EXPECT_EQ(size, images1.size());
    for (size_t i = 0; i < size; i++) {
        EXPECT_EQ(100u, images1[i]->data_.size());
        for (size_t j = 0; j < images1[i]->data_.size(); j++)
            EXPECT_EQ(ref_images1_data[i][j], images1[i]->data_[j]);
    }

    EXPECT_EQ(size, images2.size());
    for (size_t i = 0; i < size; i++) {
        EXPECT_EQ(100u, images2[i]->data_.size());
        for (size_t j = 0; j < images2[i]->data_.size(); j++)
            EXPECT_EQ(ref_images2_data[i][j], images2[i]->data_[j]);
    }
}

TEST(ColorMapOptimization, DISABLED_MakeDepthMasks) {
    vector<vector<double>> ref_images_data = {
            {0,   0,   255, 255, 255, 255, 255, 255, 255, 0,   0,   0,   255,
             255, 255, 255, 255, 255, 255, 0,   0,   0,   255, 255, 255, 255,
             255, 255, 255, 0,   0,   0,   255, 255, 255, 255, 255, 255, 255,
             0,   255, 255, 255, 255, 255, 255, 0,   0,   0,   0,   255, 255,
             255, 255, 255, 255, 0,   0,   0,   0,   255, 255, 255, 255, 255,
             255, 0,   0,   0,   0,   255, 255, 255, 255, 255, 255, 0,   0,
             0,   0,   255, 255, 255, 255, 255, 255, 0,   0,   0,   0,   255,
             255, 255, 255, 255, 255, 0,   0,   0,   0},
            {0,   255, 255, 255, 255, 255, 255, 255, 0,   0,   255, 255, 255,
             255, 255, 255, 255, 255, 0,   0,   255, 255, 255, 255, 255, 255,
             255, 255, 0,   0,   255, 255, 255, 255, 255, 255, 255, 255, 0,
             0,   255, 255, 255, 255, 255, 0,   0,   0,   0,   0,   255, 255,
             255, 255, 255, 0,   0,   0,   0,   0,   255, 255, 255, 255, 255,
             0,   0,   0,   0,   0,   255, 255, 255, 255, 255, 0,   0,   0,
             0,   0,   255, 255, 255, 255, 255, 0,   0,   0,   0,   0,   255,
             255, 255, 255, 255, 0,   0,   0,   0,   0}};

    size_t size = 2;
    int width = 10;
    int height = 10;

    vector<geometry::RGBDImage> images_rgbd =
            GenerateRGBDImages(width, height, size);

    // ColorMapOptimizationOption option(false, 62, 0.316, 30, 2.5, 0.03, 0.95,
    // 3);

    // vector<Image> images = MakeDepthMasks(images_rgbd, option);

    // EXPECT_EQ(size, images.size());
    // for (size_t i = 0; i < size; i++)
    // {
    //     EXPECT_EQ(width * height, images[i].data_.size());
    //     for (size_t j = 0; j < images[i].data_.size(); j++)
    //         EXPECT_EQ(ref_images_data[i][j], images[i].data_[j]);
    // }
}

TEST(ColorMapOptimization, DISABLED_ColorMapOptimization) {
    vector<Eigen::Vector3d> ref_vertices = {{0.000000, 0.000000, 10.000000},
                                            {0.000000, 0.000000, -10.000000},
                                            {5.877853, 0.000000, 8.090170},
                                            {4.755283, 3.454915, 8.090170},
                                            {1.816356, 5.590170, 8.090170},
                                            {-1.816356, 5.590170, 8.090170},
                                            {-4.755283, 3.454915, 8.090170},
                                            {-5.877853, 0.000000, 8.090170},
                                            {-4.755283, -3.454915, 8.090170},
                                            {-1.816356, -5.590170, 8.090170},
                                            {1.816356, -5.590170, 8.090170},
                                            {4.755283, -3.454915, 8.090170},
                                            {9.510565, 0.000000, 3.090170},
                                            {7.694209, 5.590170, 3.090170},
                                            {2.938926, 9.045085, 3.090170},
                                            {-2.938926, 9.045085, 3.090170},
                                            {-7.694209, 5.590170, 3.090170},
                                            {-9.510565, 0.000000, 3.090170},
                                            {-7.694209, -5.590170, 3.090170},
                                            {-2.938926, -9.045085, 3.090170},
                                            {2.938926, -9.045085, 3.090170},
                                            {7.694209, -5.590170, 3.090170},
                                            {9.510565, 0.000000, -3.090170},
                                            {7.694209, 5.590170, -3.090170},
                                            {2.938926, 9.045085, -3.090170},
                                            {-2.938926, 9.045085, -3.090170},
                                            {-7.694209, 5.590170, -3.090170},
                                            {-9.510565, 0.000000, -3.090170},
                                            {-7.694209, -5.590170, -3.090170},
                                            {-2.938926, -9.045085, -3.090170},
                                            {2.938926, -9.045085, -3.090170},
                                            {7.694209, -5.590170, -3.090170},
                                            {5.877853, 0.000000, -8.090170},
                                            {4.755283, 3.454915, -8.090170},
                                            {1.816356, 5.590170, -8.090170},
                                            {-1.816356, 5.590170, -8.090170},
                                            {-4.755283, 3.454915, -8.090170},
                                            {-5.877853, 0.000000, -8.090170},
                                            {-4.755283, -3.454915, -8.090170},
                                            {-1.816356, -5.590170, -8.090170},
                                            {1.816356, -5.590170, -8.090170},
                                            {4.755283, -3.454915, -8.090170}};

    vector<Eigen::Vector3d> ref_vertex_normals = {};

    vector<Eigen::Vector3d> ref_vertex_colors = {
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.000000, 0.000000, 0.000000}};

    vector<Eigen::Vector3i> ref_triangles = {
            {0, 2, 3},    {1, 33, 32},  {0, 3, 4},    {1, 34, 33},
            {0, 4, 5},    {1, 35, 34},  {0, 5, 6},    {1, 36, 35},
            {0, 6, 7},    {1, 37, 36},  {0, 7, 8},    {1, 38, 37},
            {0, 8, 9},    {1, 39, 38},  {0, 9, 10},   {1, 40, 39},
            {0, 10, 11},  {1, 41, 40},  {0, 11, 2},   {1, 32, 41},
            {12, 3, 2},   {12, 13, 3},  {13, 4, 3},   {13, 14, 4},
            {14, 5, 4},   {14, 15, 5},  {15, 6, 5},   {15, 16, 6},
            {16, 7, 6},   {16, 17, 7},  {17, 8, 7},   {17, 18, 8},
            {18, 9, 8},   {18, 19, 9},  {19, 10, 9},  {19, 20, 10},
            {20, 11, 10}, {20, 21, 11}, {21, 2, 11},  {21, 12, 2},
            {22, 13, 12}, {22, 23, 13}, {23, 14, 13}, {23, 24, 14},
            {24, 15, 14}, {24, 25, 15}, {25, 16, 15}, {25, 26, 16},
            {26, 17, 16}, {26, 27, 17}, {27, 18, 17}, {27, 28, 18},
            {28, 19, 18}, {28, 29, 19}, {29, 20, 19}, {29, 30, 20},
            {30, 21, 20}, {30, 31, 21}, {31, 12, 21}, {31, 22, 12},
            {32, 23, 22}, {32, 33, 23}, {33, 24, 23}, {33, 34, 24},
            {34, 25, 24}, {34, 35, 25}, {35, 26, 25}, {35, 36, 26},
            {36, 27, 26}, {36, 37, 27}, {37, 28, 27}, {37, 38, 28},
            {38, 29, 28}, {38, 39, 29}, {39, 30, 29}, {39, 40, 30},
            {40, 31, 30}, {40, 41, 31}, {41, 22, 31}, {41, 32, 22}};

    vector<Eigen::Vector3d> ref_triangle_normals = {};

    size_t size = 10;
    int width = 320;
    int height = 240;
    // int num_of_channels = 1;
    // int bytes_per_channel = 4;

    shared_ptr<geometry::TriangleMesh> mesh =
            geometry::TriangleMesh::CreateSphere(10.0, 5);

    vector<geometry::RGBDImage> rgbdImages =
            GenerateRGBDImages(width, height, size);

    Eigen::Vector3d pose(60, 15, 0.3);
    camera::PinholeCameraTrajectory camera =
            GenerateCamera(width, height, pose);

    // ColorMapOptimizationOption option(false, 62, 0.316, 30, 2.5, 0.03, 0.95,
    // 3);

    vector<double> proxy_intensity;

    // ColorMapOptimization(
    //     *mesh,
    //     rgbdImages,
    //     camera,
    //     option);

    EXPECT_EQ(ref_vertices.size(), mesh->vertices_.size());
    for (size_t i = 0; i < ref_vertices.size(); i++)
        ExpectEQ(ref_vertices[i], mesh->vertices_[i]);

    EXPECT_EQ(ref_vertex_normals.size(), mesh->vertex_normals_.size());
    for (size_t i = 0; i < ref_vertex_normals.size(); i++)
        ExpectEQ(ref_vertex_normals[i], mesh->vertex_normals_[i]);

    EXPECT_EQ(ref_vertex_colors.size(), mesh->vertex_colors_.size());
    for (size_t i = 0; i < ref_vertex_colors.size(); i++)
        ExpectEQ(ref_vertex_colors[i], mesh->vertex_colors_[i]);

    EXPECT_EQ(ref_triangles.size(), mesh->triangles_.size());
    for (size_t i = 0; i < ref_triangles.size(); i++)
        ExpectEQ(ref_triangles[i], mesh->triangles_[i]);

    EXPECT_EQ(ref_triangle_normals.size(), mesh->triangle_normals_.size());
    for (size_t i = 0; i < ref_triangle_normals.size(); i++)
        ExpectEQ(ref_triangle_normals[i], mesh->triangle_normals_[i]);
}
