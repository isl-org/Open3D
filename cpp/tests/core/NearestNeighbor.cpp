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

#include "open3d/core/nn/NearestNeighbor.h"

#include <cmath>
#include <limits>

#include "open3d/core/Dtype.h"
#include "open3d/core/SizeVector.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/utility/Helper.h"
#include "tests/UnitTest.h"

namespace open3d {
namespace tests {

TEST(NearestNeighbor, KnnSearch) {
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
    int knn = 30;

    std::vector<double> points;
    for (unsigned int i = 0; i < pc.points_.size(); i++) {
        Eigen::Vector3d point_vec = pc.points_[i];
        std::vector<double> point(
                point_vec.data(),
                point_vec.data() + point_vec.rows() * point_vec.cols());
        points.insert(points.end(), point.begin(), point.end());
    }

    core::Tensor ref(points.data(), {size, 3}, core::Dtype::Float64);
    core::nn::NearestNeighbor index(ref);
    index.KnnIndex();

    core::Tensor query(std::vector<double>({1.647059, 4.392157, 8.784314}),
                       {1, 3}, core::Dtype::Float64);

    std::pair<core::Tensor, core::Tensor> result = index.KnnSearch(query, knn);

    std::vector<int64_t> indices = result.first.ToFlatVector<int64_t>();
    std::vector<int> indices2(indices.begin(), indices.end());
    ExpectEQ(ref_indices, indices2);
    ExpectEQ(ref_distance2, result.second.ToFlatVector<double>());
}

TEST(NearestNeighbor, FixedRadiusSearch) {
    std::vector<int> ref_indices = {27, 48, 4,  77, 90, 7,  54, 17, 76, 38, 39,
                                    60, 15, 84, 11, 57, 3,  32, 99, 36, 52, 27,
                                    48, 4,  77, 90, 7,  54, 17, 76, 38, 39, 60,
                                    15, 84, 11, 57, 3,  32, 99, 36, 52};

    std::vector<double> ref_distance2 = {
            0.000000,  4.684353,  4.996539,  9.191849,  10.034604, 10.466745,
            10.649751, 11.434066, 12.089195, 13.345638, 13.696270, 14.016148,
            16.851978, 17.073435, 18.254518, 20.019994, 21.496347, 23.077277,
            23.692427, 23.809303, 24.104578, 0.000000,  4.684353,  4.996539,
            9.191849,  10.034604, 10.466745, 10.649751, 11.434066, 12.089195,
            13.345638, 13.696270, 14.016148, 16.851978, 17.073435, 18.254518,
            20.019994, 21.496347, 23.077277, 23.692427, 23.809303, 24.104578};

    int size = 100;

    geometry::PointCloud pc;

    Eigen::Vector3d vmin(0.0, 0.0, 0.0);
    Eigen::Vector3d vmax(10.0, 10.0, 10.0);

    pc.points_.resize(size);
    Rand(pc.points_, vmin, vmax, 0);
    double radius = 5.0;

    std::vector<double> points;
    for (unsigned int i = 0; i < pc.points_.size(); i++) {
        Eigen::Vector3d point_vec = pc.points_[i];
        std::vector<double> point(
                point_vec.data(),
                point_vec.data() + point_vec.rows() * point_vec.cols());
        points.insert(points.end(), point.begin(), point.end());
    }

    core::Tensor ref(points.data(), {size, 3}, core::Dtype::Float64);
    core::nn::NearestNeighbor index(ref);
    index.FixedRadiusIndex();

    core::Tensor query(std::vector<double>({1.647059, 4.392157, 8.784314,
                                            1.647059, 4.392157, 8.784314}),
                       {2, 3}, core::Dtype::Float64);

    std::tuple<core::Tensor, core::Tensor, core::Tensor> result =
            index.FixedRadiusSearch(query, radius);

    std::vector<int64_t> indices = std::get<0>(result).ToFlatVector<int64_t>();
    std::vector<int> indices2(indices.begin(), indices.end());
    ExpectEQ(ref_indices, indices2);
    ExpectEQ(ref_distance2, std::get<1>(result).ToFlatVector<double>());
}

TEST(NearestNeighbor, RadiusSearch) {
    std::vector<int> ref_indices = {27, 48, 4,  77, 90, 7,  54, 17, 76, 38, 39,
                                    60, 15, 84, 11, 57, 3,  32, 99, 36, 52, 27,
                                    48, 4,  77, 90, 7,  54, 17, 76, 38, 39, 60,
                                    15, 84, 11, 57, 3,  32, 99, 36, 52};

    std::vector<double> ref_distance2 = {
            0.000000,  4.684353,  4.996539,  9.191849,  10.034604, 10.466745,
            10.649751, 11.434066, 12.089195, 13.345638, 13.696270, 14.016148,
            16.851978, 17.073435, 18.254518, 20.019994, 21.496347, 23.077277,
            23.692427, 23.809303, 24.104578, 0.000000,  4.684353,  4.996539,
            9.191849,  10.034604, 10.466745, 10.649751, 11.434066, 12.089195,
            13.345638, 13.696270, 14.016148, 16.851978, 17.073435, 18.254518,
            20.019994, 21.496347, 23.077277, 23.692427, 23.809303, 24.104578};

    int size = 100;
    double radii[] = {5.0, 5.0};

    geometry::PointCloud pc;

    Eigen::Vector3d vmin(0.0, 0.0, 0.0);
    Eigen::Vector3d vmax(10.0, 10.0, 10.0);

    pc.points_.resize(size);
    Rand(pc.points_, vmin, vmax, 0);

    std::vector<double> points;
    for (unsigned int i = 0; i < pc.points_.size(); i++) {
        Eigen::Vector3d point_vec = pc.points_[i];
        std::vector<double> point(
                point_vec.data(),
                point_vec.data() + point_vec.rows() * point_vec.cols());
        points.insert(points.end(), point.begin(), point.end());
    }

    core::Tensor ref(points.data(), {size, 3}, core::Dtype::Float64);
    core::nn::NearestNeighbor index(ref);
    index.RadiusIndex();

    core::Tensor query(std::vector<double>({1.647059, 4.392157, 8.784314,
                                            1.647059, 4.392157, 8.784314}),
                       {2, 3}, core::Dtype::Float64);

    std::tuple<core::Tensor, core::Tensor, core::Tensor> result =
            index.RadiusSearch(query, radii);

    std::vector<int64_t> indices = std::get<0>(result).ToFlatVector<int64_t>();
    std::vector<int> indices2(indices.begin(), indices.end());
    ExpectEQ(ref_indices, indices2);
    ExpectEQ(ref_distance2, std::get<1>(result).ToFlatVector<double>());
}

TEST(NearestNeighbor, HybridSearch) {
    std::vector<int> ref_indices = {27, 48, 4, -1, -1};

    std::vector<double> ref_distance = {0.000000, 4.684353, 4.996539, 0.0, 0.0};

    int size = 100;
    double radius = 5.0;
    int max_knn = 5;

    geometry::PointCloud pc;

    Eigen::Vector3d vmin(0.0, 0.0, 0.0);
    Eigen::Vector3d vmax(10.0, 10.0, 10.0);

    pc.points_.resize(size);
    Rand(pc.points_, vmin, vmax, 0);

    std::vector<double> points;
    for (unsigned int i = 0; i < pc.points_.size(); i++) {
        Eigen::Vector3d point_vec = pc.points_[i];
        std::vector<double> point(
                point_vec.data(),
                point_vec.data() + point_vec.rows() * point_vec.cols());
        points.insert(points.end(), point.begin(), point.end());
    }

    core::Tensor ref(points.data(), {size, 3}, core::Dtype::Float64);
    core::nn::NearestNeighbor index(ref);
    index.HybridIndex();

    core::Tensor query(std::vector<double>({
                               1.647059,
                               4.392157,
                               8.784314,
                       }),
                       {1, 3}, core::Dtype::Float64);

    std::pair<core::Tensor, core::Tensor> result =
            index.HybridSearch(query, radius, max_knn);

    std::vector<int64_t> indices = result.first.ToFlatVector<int64_t>();
    std::vector<int> indices2(indices.begin(), indices.end());
    ExpectEQ(ref_indices, indices2);
    ExpectEQ(ref_distance, result.second.ToFlatVector<double>());
}

}  // namespace tests
}  // namespace open3d
