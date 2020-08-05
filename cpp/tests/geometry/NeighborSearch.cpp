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
#include "open3d/geometry/NeighborSearch.h"
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

TEST(NeighborSearch, SearchKNN) {
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

    geometry::NeighborSearch neighborsearch(t);

    vector<float> query_vec = {1.647059, 4.392157, 8.784314};
    core::Tensor q(query_vec, {1, 3}, core::Dtype::Float32, device);
    int knn = 30;
    vector<long> indices;
    vector<float> distance2;

    pair<core::Tensor, core::Tensor> result = neighborsearch.SearchKNN(q, knn);

    indices = result.first.ToFlatVector<long>();
    distance2 = result.second.ToFlatVector<float>();
    vector<int> indices2(indices.begin(), indices.end());

    ExpectEQ(ref_indices, indices2);
    ExpectEQ(ref_distance2, distance2, threshold);
}

TEST(NeighborSearch, SearchKNN_MultiQuery) {
    vector<int> ref_indices = {27, 48, 4,  77, 90, 7,  54, 17, 76, 38,
                               39, 60, 15, 84, 11, 
                               28, 94, 92, 33, 18, 68, 44, 58, 21, 23, 
                               80, 87, 64, 35, 89};

    vector<float> ref_distance2 = {
            0.000000,  4.684353,  4.996539,  9.191849,  10.034604, 10.466745,
            10.649751, 11.434066, 12.089195, 13.345638, 13.696270, 14.016148,
            16.851978, 17.073435, 18.254518, 
            0.000000,  0.187620,  1.527110,  3.031130,  3.131100,  5.7424, 
            7.271060,  8.319890,  10.08070,  10.12840,  10.97730,  12.52132, 
            13.25340,  14.01610,  14.56210};

    int size = 100;
    float threshold = 5e-5;

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

    geometry::NeighborSearch neighborsearch(t);

    vector<float> query_vec = {1.647059, 4.392157, 8.784314,
                               8.274510, 3.294120, 2.274510};
    core::Tensor q(query_vec, {2, 3}, core::Dtype::Float32, device);
    int knn = 15;
    vector<long> indices;
    vector<float> distance2;

    pair<core::Tensor, core::Tensor> result = neighborsearch.SearchKNN(q, knn);

    indices = result.first.ToFlatVector<long>();
    distance2 = result.second.ToFlatVector<float>();
    vector<int> indices2(indices.begin(), indices.end());

    ExpectEQ(ref_indices, indices2);
    ExpectEQ(ref_distance2, distance2, threshold);
}

TEST(NeighborSearch, SearchKNN_2e6) {
    vector<int> ref_indices = {27,      1152930, 1224678, 1913298, 241116, 1006205, 1059237, 1714823, 1074643, 238765, 
                               1346254, 937272,  1766692, 401566,  544783, 1003373, 1925626, 1217426, 366060,  1054994, 
                               1808440, 494538,  1483669, 1982240, 324281, 1504383, 1767616, 41418,   446301,  646945};

    vector<float> ref_distance2 = {
            0.00000, 0.25552, 0.27129, 0.38407, 0.45310, 0.65057, 0.78246, 0.96896, 0.98150, 1.04516, 
            1.15342, 1.16118, 1.31195, 1.39012, 1.48130, 1.48707, 1.49334, 1.50074, 1.52727, 1.59777, 
            1.76207, 1.77287, 1.79251, 1.79462, 1.83828, 1.89121, 1.92284, 2.00621, 2.10253, 2.12210};

    srand(0);
    unsigned int size = 2e6;
    float threshold = 1e-5;

    std::vector<float> points;
    points.resize(2e6 * 3);
    for (unsigned int i = 0; i < size; i++) {
        for (unsigned int j = 0; j < 3; j++) {
            points[3 * i + j] = ((float)rand() / RAND_MAX) * 100.0;
        }
    }

    core::Device device;
    core::Tensor t(points, {(long)2e6, 3}, core::Dtype::Float32, device);

    geometry::NeighborSearch neighborsearch(t);

    vector<float> query_vec = {16.5974, 44.0105, 88.0075};
    core::Tensor q(query_vec, {1, 3}, core::Dtype::Float32, device);
    int knn = 30;
    vector<long> indices;
    vector<float> distance2;

    pair<core::Tensor, core::Tensor> result = neighborsearch.SearchKNN(q, knn);

    indices = result.first.ToFlatVector<long>();
    distance2 = result.second.ToFlatVector<float>();
    vector<int> indices2(indices.begin(), indices.end());

    ExpectEQ(ref_indices, indices2);
    ExpectEQ(ref_distance2, distance2, threshold);
}

TEST(NeighborSearch, SearchKNN_GPU_MultiQuery) {
    vector<int> ref_indices = {27, 48, 4,  77, 90, 7,  54, 17, 76, 38,
                               39, 60, 15, 84, 11, 
                               28, 94, 92, 33, 18, 68, 44, 58, 21, 23, 
                               80, 87, 64, 35, 89};

    vector<float> ref_distance2 = {
            0.000000,  4.684353,  4.996539,  9.191849,  10.034604, 10.466745,
            10.649751, 11.434066, 12.089195, 13.345638, 13.696270, 14.016148,
            16.851978, 17.073435, 18.254518, 
            0.000000,  0.187620,  1.527110,  3.031130,  3.131100,  5.7424, 
            7.271060,  8.319890,  10.08070,  10.12840,  10.97730,  12.52132, 
            13.25340,  14.01610,  14.56210};

    int size = 100;
    float threshold = 5e-5;

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

    geometry::NeighborSearch neighborsearch(t);

    vector<float> query_vec = {1.647059, 4.392157, 8.784314,
                               8.274510, 3.294120, 2.274510};
    core::Tensor q(query_vec, {2, 3}, core::Dtype::Float32, device);
    int knn = 15;
    vector<long> indices;
    vector<float> distance2;

    pair<core::Tensor, core::Tensor> result = neighborsearch.SearchKNN(q, knn);

    indices = result.first.ToFlatVector<long>();
    distance2 = result.second.ToFlatVector<float>();
    vector<int> indices2(indices.begin(), indices.end());

    ExpectEQ(ref_indices, indices2);
    ExpectEQ(ref_distance2, distance2, threshold);
}

TEST(NeighborSearch, SearchHybrid) {
    vector<int> ref_indices = {27, 48, 4,  77, 90, 7,  54, 17, -1, -1,
                               -1, -1, -1, -1, -1, 
                               28, 94, 92, 33, 18, 68, 44, 58, 21, 23, 
                               80, -1, -1, -1, -1};

    vector<float> ref_distance2 = {
            0.000000,  4.684353,  4.996539, 9.191849, 10.034604, 10.466745,
            10.649751, 11.434066, 0.000000, 0.000000, 0.000000,  0.000000,
            0.000000,  0.000000,  0.000000, 
            0.000000,  0.187620,  1.527110,  3.031130,  3.131100,  5.7424, 
            7.271060,  8.319890,  10.08070,  10.12840,  10.97730,  0.000000, 
            0.000000,  0.000000,  0.000000};

    int size = 100;
    float threshold = 5e-4;

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

    geometry::NeighborSearch neighborsearch(t);

    vector<float> query_vec = {1.647059, 4.392157, 8.784314,
                               8.274510, 3.294120, 2.274510};
    core::Tensor q(query_vec, {2, 3}, core::Dtype::Float32, device);
    int max_nn = 15;
    float radius = 12;
    vector<long> indices;
    vector<float> distance2;

    pair<core::Tensor, core::Tensor> result = neighborsearch.SearchHybrid(q, radius, max_nn);

    indices = result.first.ToFlatVector<long>();
    distance2 = result.second.ToFlatVector<float>();
    vector<int> indices2(indices.begin(), indices.end());

    ExpectEQ(ref_indices, indices2);
    ExpectEQ(ref_distance2, distance2, threshold);
}

}  // namespace tests
}  // namespace open3d
