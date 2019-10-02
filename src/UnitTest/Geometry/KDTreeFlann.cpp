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

#include "Open3D/Geometry/KDTreeFlann.h"
#include "Open3D/Geometry/PointCloud.h"
#include "Open3D/Geometry/TriangleMesh.h"
#include "TestUtility/UnitTest.h"

using namespace Eigen;
using namespace open3d;
using namespace std;
using namespace unit_test;

TEST(KDTreeFlann, DISABLED_Search) { unit_test::NotImplemented(); }

TEST(KDTreeFlann, SearchKNN) {
    vector<int> ref_indices = {27, 48, 4,  77, 90, 7,  54, 17, 76, 38,
                               39, 60, 15, 84, 11, 57, 3,  32, 99, 36,
                               52, 40, 26, 59, 22, 97, 20, 42, 73, 24};

    vector<double> ref_distance2 = {
            0.000000,  4.684353,  4.996539,  9.191849,  10.034604, 10.466745,
            10.649751, 11.434066, 12.089195, 13.345638, 13.696270, 14.016148,
            16.851978, 17.073435, 18.254518, 20.019994, 21.496347, 23.077277,
            23.692427, 23.809303, 24.104578, 25.005770, 26.952710, 27.487888,
            27.998463, 28.262975, 28.581313, 28.816608, 31.603230, 31.610916};

    int size = 100;

    geometry::PointCloud pc;

    Vector3d vmin(0.0, 0.0, 0.0);
    Vector3d vmax(10.0, 10.0, 10.0);

    pc.points_.resize(size);
    Rand(pc.points_, vmin, vmax, 0);

    geometry::KDTreeFlann kdtree(pc);

    Vector3d query = {1.647059, 4.392157, 8.784314};
    int knn = 30;
    vector<int> indices;
    vector<double> distance2;

    int result = kdtree.SearchKNN(query, knn, indices, distance2);

    EXPECT_EQ(result, 30);

    ExpectEQ(ref_indices, indices);
    ExpectEQ(ref_distance2, distance2);
}

TEST(KDTreeFlann, SearchRadius) {
    vector<int> ref_indices = {27, 48, 4,  77, 90, 7, 54, 17, 76, 38, 39,
                               60, 15, 84, 11, 57, 3, 32, 99, 36, 52};

    vector<double> ref_distance2 = {
            0.000000,  4.684353,  4.996539,  9.191849,  10.034604, 10.466745,
            10.649751, 11.434066, 12.089195, 13.345638, 13.696270, 14.016148,
            16.851978, 17.073435, 18.254518, 20.019994, 21.496347, 23.077277,
            23.692427, 23.809303, 24.104578};

    int size = 100;

    geometry::PointCloud pc;

    Vector3d vmin(0.0, 0.0, 0.0);
    Vector3d vmax(10.0, 10.0, 10.0);

    pc.points_.resize(size);
    Rand(pc.points_, vmin, vmax, 0);

    geometry::KDTreeFlann kdtree(pc);

    Vector3d query = {1.647059, 4.392157, 8.784314};
    double radius = 5.0;
    vector<int> indices;
    vector<double> distance2;

    int result =
            kdtree.SearchRadius<Vector3d>(query, radius, indices, distance2);

    EXPECT_EQ(result, 21);

    ExpectEQ(ref_indices, indices);
    ExpectEQ(ref_distance2, distance2);
}

TEST(KDTreeFlann, SearchHybrid) {
    vector<int> ref_indices = {27, 48, 4,  77, 90, 7,  54, 17,
                               76, 38, 39, 60, 15, 84, 11};

    vector<double> ref_distance2 = {0.000000,  4.684353,  4.996539,  9.191849,
                                    10.034604, 10.466745, 10.649751, 11.434066,
                                    12.089195, 13.345638, 13.696270, 14.016148,
                                    16.851978, 17.073435, 18.254518};

    int size = 100;

    geometry::PointCloud pc;

    Vector3d vmin(0.0, 0.0, 0.0);
    Vector3d vmax(10.0, 10.0, 10.0);

    pc.points_.resize(size);
    Rand(pc.points_, vmin, vmax, 0);

    geometry::KDTreeFlann kdtree(pc);

    Vector3d query = {1.647059, 4.392157, 8.784314};
    int max_nn = 15;
    double radius = 5.0;
    vector<int> indices;
    vector<double> distance2;

    int result = kdtree.SearchHybrid<Vector3d>(query, radius, max_nn, indices,
                                               distance2);

    EXPECT_EQ(result, 15);

    ExpectEQ(ref_indices, indices);
    ExpectEQ(ref_distance2, distance2);
}
