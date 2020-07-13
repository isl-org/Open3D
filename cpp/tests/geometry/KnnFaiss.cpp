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
#include "open3d/geometry/KnnFaiss.h"
#include "open3d/core/Device.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/TriangleMesh.h"

#include "core/CoreTest.h"
#include "tests/UnitTest.h"

using namespace Eigen;
using namespace open3d;
using namespace std;

namespace open3d {
namespace tests {

TEST(KnnFaiss, DISABLED_Search) { NotImplemented(); }

TEST(KnnFaiss, SearchKNN) {
    vector<int> ref_indices = {27, 48, 4,  77, 90, 7,  54, 17, 76, 38,
                               39, 60, 15, 84, 11, 57, 3,  32, 99, 36,
                               52, 40, 26, 59, 22, 97, 20, 42, 73, 24};

    vector<float> ref_distance2 = {
            0.000000,  4.684353,  4.996539,  9.191849,  10.034604, 10.466745,
            10.649751, 11.434066, 12.089195, 13.345638, 13.696270, 14.016148,
            16.851978, 17.073435, 18.254518, 20.019994, 21.496347, 23.077277,
            23.692427, 23.809303, 24.104578, 25.005770, 26.952710, 27.487888,
            27.998463, 28.262975, 28.581313, 28.816608, 31.603230, 31.610916};

    int size = 100;
    float threshold = 1e-5;

    geometry::PointCloud pc;

    Vector3d vmin(0.0, 0.0, 0.0);
    Vector3d vmax(10.0, 10.0, 10.0);

    pc.points_.resize(size);
    Rand(pc.points_, vmin, vmax, 0);

    geometry::KnnFaiss knnFaiss(pc);

    Vector3d query = {1.647059, 4.392157, 8.784314};
    int knn = 30;
    vector<long> indices;
    vector<float> distance2;

    int result = knnFaiss.SearchKNN(query, knn, indices, distance2);

    vector<int> indices2(indices.begin(), indices.end());

    EXPECT_EQ(result, 30);

    ExpectEQ(ref_indices, indices2);
    ExpectEQ(ref_distance2, distance2, threshold);
}

TEST(KnnFaiss, SearchRadius) {
    vector<int> ref_indices = {27, 48, 4,  77, 90, 7, 54, 17, 76, 38, 39,
                               60, 15, 84, 11, 57, 3, 32, 99, 36, 52};

    vector<double> ref_distance2 = {
            0.000000,  4.684353,  4.996539,  9.191849,  10.034604, 10.466745,
            10.649751, 11.434066, 12.089195, 13.345638, 13.696270, 14.016148,
            16.851978, 17.073435, 18.254518, 20.019994, 21.496347, 23.077277,
            23.692427, 23.809303, 24.104578};

    int size = 100;
    float threshold = 1e-5;

    geometry::PointCloud pc;

    Vector3d vmin(0.0, 0.0, 0.0);
    Vector3d vmax(10.0, 10.0, 10.0);

    pc.points_.resize(size);
    Rand(pc.points_, vmin, vmax, 0);

    geometry::KnnFaiss knnFaiss(pc);

    Vector3d query = {1.647059, 4.392157, 8.784314};
    float radius = 5.0;
    vector<long> indices;
    vector<float> distance2;

    int result =
            knnFaiss.SearchRadius<Vector3d>(query, radius, indices, distance2);

    vector<int> indices2(indices.begin(), indices.end());
    vector<double> distance3(distance2.begin(), distance2.end());

    EXPECT_EQ(result, 21);

    ExpectEQ(ref_indices, indices2);
    ExpectEQ(ref_distance2, distance3, threshold);
}

TEST(KnnFaiss, DISABLED_SearchHybrid) { NotImplemented(); }

TEST(KnnFaiss, SetTensorData) {
    vector<int> ref_indices = {27, 48, 4,  77, 90, 7,  54, 17, 76, 38,
                               39, 60, 15, 84, 11, 57, 3,  32, 99, 36,
                               52, 40, 26, 59, 22, 97, 20, 42, 73, 24};

    vector<float> ref_distance2 = {
            0.000000,  4.684353,  4.996539,  9.191849,  10.034604, 10.466745,
            10.649751, 11.434066, 12.089195, 13.345638, 13.696270, 14.016148,
            16.851978, 17.073435, 18.254518, 20.019994, 21.496347, 23.077277,
            23.692427, 23.809303, 24.104578, 25.005770, 26.952710, 27.487888,
            27.998463, 28.262975, 28.581313, 28.816608, 31.603230, 31.610916};

    int size = 100;
    float threshold = 1e-5;

    geometry::PointCloud pc;

    Vector3d vmin(0.0, 0.0, 0.0);
    Vector3d vmax(10.0, 10.0, 10.0);

    pc.points_.resize(size);
    Rand(pc.points_, vmin, vmax, 0);

    std::vector<float> points;
    points.resize(100 * 3);
    for (unsigned int i = 0; i < pc.points_.size(); i++) {
        for (unsigned int j = 0; j < 3; j++) {
            points[3 * i + j] = (float)pc.points_[i].data()[j];
        }
    }

    core::Device device;
    core::Tensor t(points, {100, 3}, core::Dtype::Float32, device);

    geometry::KnnFaiss knnFaiss(t);

    Vector3d query = {1.647059, 4.392157, 8.784314};
    int knn = 30;
    vector<long> indices;
    vector<float> distance2;

    int result = knnFaiss.SearchKNN(query, knn, indices, distance2);

    vector<int> indices2(indices.begin(), indices.end());

    EXPECT_EQ(result, 30);

    ExpectEQ(ref_indices, indices2);
    ExpectEQ(ref_distance2, distance2, threshold);
}
TEST(KnnFaiss, SetTensorData_GPU) {
    std::cout << "Test" << std::endl;
    vector<int> ref_indices = {27, 48, 4,  77, 90, 7,  54, 17, 76, 38,
                               39, 60, 15, 84, 11, 57, 3,  32, 99, 36,
                               52, 40, 26, 59, 22, 97, 20, 42, 73, 24};

    vector<float> ref_distance2 = {
            0.000000,  4.684353,  4.996539,  9.191849,  10.034604, 10.466745,
            10.649751, 11.434066, 12.089195, 13.345638, 13.696270, 14.016148,
            16.851978, 17.073435, 18.254518, 20.019994, 21.496347, 23.077277,
            23.692427, 23.809303, 24.104578, 25.005770, 26.952710, 27.487888,
            27.998463, 28.262975, 28.581313, 28.816608, 31.603230, 31.610916};

    int size = 100;
    float threshold = 1e-5;

    geometry::PointCloud pc;

    Vector3d vmin(0.0, 0.0, 0.0);
    Vector3d vmax(10.0, 10.0, 10.0);

    pc.points_.resize(size);
    Rand(pc.points_, vmin, vmax, 0);

    std::vector<float> points;
    points.resize(100 * 3);
    for (unsigned int i = 0; i < pc.points_.size(); i++) {
        for (unsigned int j = 0; j < 3; j++) {
            points[3 * i + j] = (float)pc.points_[i].data()[j];
        }
    }

    core::Device device("CUDA:0");
    core::Tensor t(points, {100, 3}, core::Dtype::Float32, device);
    geometry::KnnFaiss knnFaiss(t);

    Vector3d query = {1.647059, 4.392157, 8.784314};
    int knn = 30;
    vector<long> indices;
    vector<float> distance2;

    int result = knnFaiss.SearchKNN(query, knn, indices, distance2);

    vector<int> indices2(indices.begin(), indices.end());

    EXPECT_EQ(result, 30);

    ExpectEQ(ref_indices, indices2);
    ExpectEQ(ref_distance2, distance2, threshold);
}

}  // namespace tests
}  // namespace open3d
