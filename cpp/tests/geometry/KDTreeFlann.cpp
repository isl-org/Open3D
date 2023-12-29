// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/geometry/KDTreeFlann.h"

#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/TriangleMesh.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

TEST(KDTreeFlann, DISABLED_Search) { NotImplemented(); }

TEST(KDTreeFlann, SearchKNN) {
    std::vector<int> ref_indices = {27, 48, 4,  77, 90, 7,  54, 17, 76, 38,
                                    39, 60, 15, 84, 11, 57, 3,  32, 99, 36,
                                    52, 40, 26, 59, 22, 97, 20, 42, 73, 24};

    std::vector<double> ref_distance2 = {
            0.000000,  4.684353,  4.996539,  9.191849,  10.034604, 10.466745,
            10.649751, 11.434066, 12.089195, 13.345638, 13.696270, 14.016148,
            16.851978, 17.073435, 18.254518, 20.019994, 21.496347, 23.077277,
            23.692427, 23.809303, 24.104578, 25.005770, 26.952710, 27.487888,
            27.998463, 28.262975, 28.581313, 28.816608, 31.603230, 31.610916};

    int size = 100;

    geometry::PointCloud pc;

    Eigen::Vector3d vmin(0.0, 0.0, 0.0);
    Eigen::Vector3d vmax(10.0, 10.0, 10.0);

    pc.points_.resize(size);
    Rand(pc.points_, vmin, vmax, 0);

    geometry::KDTreeFlann kdtree(pc);

    Eigen::Vector3d query = {1.647059, 4.392157, 8.784314};
    int knn = 30;
    std::vector<int> indices;
    std::vector<double> distance2;

    int result = kdtree.SearchKNN(query, knn, indices, distance2);

    EXPECT_EQ(result, 30);

    ExpectEQ(ref_indices, indices);
    ExpectEQ(ref_distance2, distance2);
}

TEST(KDTreeFlann, SearchRadius) {
    std::vector<int> ref_indices = {27, 48, 4,  77, 90, 7, 54, 17, 76, 38, 39,
                                    60, 15, 84, 11, 57, 3, 32, 99, 36, 52};

    std::vector<double> ref_distance2 = {
            0.000000,  4.684353,  4.996539,  9.191849,  10.034604, 10.466745,
            10.649751, 11.434066, 12.089195, 13.345638, 13.696270, 14.016148,
            16.851978, 17.073435, 18.254518, 20.019994, 21.496347, 23.077277,
            23.692427, 23.809303, 24.104578};

    int size = 100;

    geometry::PointCloud pc;

    Eigen::Vector3d vmin(0.0, 0.0, 0.0);
    Eigen::Vector3d vmax(10.0, 10.0, 10.0);

    pc.points_.resize(size);
    Rand(pc.points_, vmin, vmax, 0);

    geometry::KDTreeFlann kdtree(pc);

    Eigen::Vector3d query = {1.647059, 4.392157, 8.784314};
    double radius = 5.0;
    std::vector<int> indices;
    std::vector<double> distance2;

    int result = kdtree.SearchRadius<Eigen::Vector3d>(query, radius, indices,
                                                      distance2);

    EXPECT_EQ(result, 21);

    ExpectEQ(ref_indices, indices);
    ExpectEQ(ref_distance2, distance2);
}

TEST(KDTreeFlann, SearchHybrid) {
    std::vector<int> ref_indices = {27, 48, 4,  77, 90, 7,  54, 17,
                                    76, 38, 39, 60, 15, 84, 11};

    std::vector<double> ref_distance2 = {
            0.000000,  4.684353,  4.996539,  9.191849,  10.034604,
            10.466745, 10.649751, 11.434066, 12.089195, 13.345638,
            13.696270, 14.016148, 16.851978, 17.073435, 18.254518};

    int size = 100;

    geometry::PointCloud pc;

    Eigen::Vector3d vmin(0.0, 0.0, 0.0);
    Eigen::Vector3d vmax(10.0, 10.0, 10.0);

    pc.points_.resize(size);
    Rand(pc.points_, vmin, vmax, 0);

    geometry::KDTreeFlann kdtree(pc);

    Eigen::Vector3d query = {1.647059, 4.392157, 8.784314};
    int max_nn = 15;
    double radius = 5.0;
    std::vector<int> indices;
    std::vector<double> distance2;

    int result = kdtree.SearchHybrid<Eigen::Vector3d>(query, radius, max_nn,
                                                      indices, distance2);

    EXPECT_EQ(result, 15);

    ExpectEQ(ref_indices, indices);
    ExpectEQ(ref_distance2, distance2);
}

}  // namespace tests
}  // namespace open3d
