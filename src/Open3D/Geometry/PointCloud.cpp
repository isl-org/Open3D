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

#include "PointCloud.h"

#include <Eigen/Dense>
#include <Open3D/Utility/Console.h>
#include <Open3D/Geometry/KDTreeFlann.h>

#include <iostream>
#include <iomanip>
using namespace std;

#ifdef OPEN3D_USE_CUDA
#include "Open3D/Utility/CUDA.cuh"
#endif
#include "Open3D/Types/Mat.h"

namespace open3d {
namespace geometry {

void PointCloud::Clear() {
    points_.clear();
    normals_.clear();
    colors_.clear();
}

bool PointCloud::IsEmpty() const { return !HasPoints(); }

Eigen::Vector3d PointCloud::GetMinBound() const {
    if (!HasPoints()) return Eigen::Vector3d::Zero();

#ifdef OPEN3D_USE_CUDA
    if (cuda::DeviceID::CPU == points_.device_id)
        return GetMinBoundCPU();
    else
        return GetMinBoundGPU();
#else
    return GetMinBoundCPU();
#endif
}

Eigen::Vector3d PointCloud::GetMaxBound() const {
    if (!HasPoints()) return Eigen::Vector3d::Zero();

#ifdef OPEN3D_USE_CUDA
    if (cuda::DeviceID::CPU == points_.device_id)
        return GetMaxBoundCPU();
    else
        return GetMaxBoundGPU();
#else
    return GetMaxBoundCPU();
#endif
}

Eigen::Vector3d PointCloud::GetMinBoundCPU() const {
    auto itr_x = std::min_element(
            points_.begin(), points_.end(),
            [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) {
                return a(0) < b(0);
            });
    auto itr_y = std::min_element(
            points_.begin(), points_.end(),
            [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) {
                return a(1) < b(1);
            });
    auto itr_z = std::min_element(
            points_.begin(), points_.end(),
            [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) {
                return a(2) < b(2);
            });

    return Eigen::Vector3d((*itr_x)(0), (*itr_y)(1), (*itr_z)(2));
}

Eigen::Vector3d PointCloud::GetMaxBoundCPU() const {
    auto itr_x = std::max_element(
            points_.begin(), points_.end(),
            [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) {
                return a(0) < b(0);
            });
    auto itr_y = std::max_element(
            points_.begin(), points_.end(),
            [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) {
                return a(1) < b(1);
            });
    auto itr_z = std::max_element(
            points_.begin(), points_.end(),
            [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) {
                return a(2) < b(2);
            });
    return Eigen::Vector3d((*itr_x)(0), (*itr_y)(1), (*itr_z)(2));
}

#ifdef OPEN3D_USE_CUDA

Eigen::Vector3d PointCloud::GetMinBoundGPU() const {
    size_t num_elements = points_.size();
    double *data = points_.d_data;
    cuda::DeviceID::Type device_id = points_.device_id;
    vector<open3d::Vec3d> minBounds(num_elements);
    double *output = (double*)minBounds.data();

    cudaError_t status = cudaSuccess;

    // get lower bounds on GPU
    status = getMinBoundHelper(device_id, data, num_elements, output);
    cuda::DebugInfo("GetMinBoundGPU:01", status);
    if (cudaSuccess != status) return Eigen::Vector3d::Zero();

    auto it = std::min_element(minBounds.begin(), minBounds.end());

    Eigen::Vector3d minBound((*it)[0], (*it)[1], (*it)[2]);

    return minBound;
}

Eigen::Vector3d PointCloud::GetMaxBoundGPU() const {
    size_t num_elements = points_.size();
    double *data = points_.d_data;
    cuda::DeviceID::Type device_id = points_.device_id;
    vector<open3d::Vec3d> maxBounds(num_elements);
    double *output = (double*)maxBounds.data();

    cudaError_t status = cudaSuccess;

    // get lower bounds on GPU
    status = getMinBoundHelper(device_id, data, num_elements, output);
    cuda::DebugInfo("GetMaxBoundGPU:01", status);
    if (cudaSuccess != status) return Eigen::Vector3d::Zero();

    auto it = std::max_element(maxBounds.begin(), maxBounds.end());

    Eigen::Vector3d maxBound((*it)[0], (*it)[1], (*it)[2]);

    return maxBound;
}

#endif

void PointCloud::Transform(const Eigen::Matrix4d &transformation) {
#ifdef OPEN3D_USE_CUDA
    if (cuda::DeviceID::CPU == points_.device_id)
        return TransformCPU(transformation);
    else
        return TransformGPU(transformation);
#else
    return TransformCPU(transformation);
#endif
}

void PointCloud::TransformCPU(const Eigen::Matrix4d &transformation) {
    for (auto &point : points_) {
        Eigen::Vector4d new_point =
                transformation *
                Eigen::Vector4d(point(0), point(1), point(2), 1.0);
        point = new_point.block<3, 1>(0, 0);
    }
    for (auto &normal : normals_) {
        Eigen::Vector4d new_normal =
                transformation *
                Eigen::Vector4d(normal(0), normal(1), normal(2), 0.0);
        normal = new_normal.block<3, 1>(0, 0);
    }
}

#ifdef OPEN3D_USE_CUDA

void PointCloud::TransformGPU(const Eigen::Matrix4d &transformation) {
    size_t num_elements = 0;
    double *data = NULL;

    // Note: this expects that both the points_ and the normals_ are on the GPU
    cuda::DeviceID::Type device_id = points_.device_id;
    open3d::Mat4d t{};
    open3d::Vec4d c{};

    for (size_t r = 0; r < t.Rows; r++)
        for (size_t c = 0; c < t.Cols; c++) t[r][c] = transformation(r, c);

    cudaError_t status = cudaSuccess;

    // transform points_ on GPU
    num_elements = points_.size();
    data = points_.d_data;
    c = { 1.0, 1.0, 1.0, 1.0 };
    status = transformHelper(device_id, data, num_elements, t, c);
    cuda::DebugInfo("TransformGPU:01", status);
    if (cudaSuccess != status) return;

    // transform normals_ on GPU
    num_elements = normals_.size();
    data = normals_.d_data;
    c = { 0.0, 0.0, 0.0, 0.0 };
    status = transformHelper(device_id, data, num_elements, t, c);
    cuda::DebugInfo("TransformGPU:02", status);
    if (cudaSuccess != status) return;
}

#endif

PointCloud &PointCloud::operator+=(const PointCloud &cloud) {
    // We do not use std::vector::insert to combine std::vector because it will
    // crash if the pointcloud is added to itself.
    if (cloud.IsEmpty()) return (*this);
    size_t old_vert_num = points_.size();
    size_t add_vert_num = cloud.points_.size();
    size_t new_vert_num = old_vert_num + add_vert_num;
    if ((!HasPoints() || HasNormals()) && cloud.HasNormals()) {
        normals_.resize(new_vert_num);
        for (size_t i = 0; i < add_vert_num; i++)
            normals_[old_vert_num + i] = cloud.normals_[i];
    } else {
        normals_.clear();
    }
    if ((!HasPoints() || HasColors()) && cloud.HasColors()) {
        colors_.resize(new_vert_num);
        for (size_t i = 0; i < add_vert_num; i++)
            colors_[old_vert_num + i] = cloud.colors_[i];
    } else {
        colors_.clear();
    }
    points_.resize(new_vert_num);
    for (size_t i = 0; i < add_vert_num; i++)
        points_[old_vert_num + i] = cloud.points_[i];
    return (*this);
}

PointCloud PointCloud::operator+(const PointCloud &cloud) const {
    return (PointCloud(*this) += cloud);
}

std::vector<double> ComputePointCloudToPointCloudDistance(
        const PointCloud &source, const PointCloud &target) {
    std::vector<double> distances(source.points_.size());
    KDTreeFlann kdtree;
    kdtree.SetGeometry(target);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < (int)source.points_.size(); i++) {
        std::vector<int> indices(1);
        std::vector<double> dists(1);
        if (kdtree.SearchKNN(source.points_[i], 1, indices, dists) == 0) {
            utility::PrintDebug(
                    "[ComputePointCloudToPointCloudDistance] Found a point "
                    "without neighbors.\n");
            distances[i] = 0.0;
        } else {
            distances[i] = std::sqrt(dists[0]);
        }
    }
    return distances;
}

std::tuple<Eigen::Vector3d, Eigen::Matrix3d> ComputePointCloudMeanAndCovariance(
        PointCloud &input) {
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity();

    if (input.IsEmpty()) return std::make_tuple(mean, covariance);

#ifdef OPEN3D_USE_CUDA
    if (cuda::DeviceID::CPU == input.points_.device_id)
        return ComputePointCloudMeanAndCovarianceCPU(input);
    else
        return ComputePointCloudMeanAndCovarianceGPU(input);
#else
    return ComputePointCloudMeanAndCovarianceCPU(input);
#endif
}

std::tuple<Eigen::Vector3d, Eigen::Matrix3d>
ComputePointCloudMeanAndCovarianceCPU(const PointCloud &input) {
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity();

    if (input.IsEmpty()) return std::make_tuple(mean, covariance);

    Eigen::Matrix<double, 9, 1> cumulants;
    cumulants.setZero();
    for (const auto &point : input.points_) {
        cumulants(0) += point(0);
        cumulants(1) += point(1);
        cumulants(2) += point(2);
        cumulants(3) += point(0) * point(0);
        cumulants(4) += point(0) * point(1);
        cumulants(5) += point(0) * point(2);
        cumulants(6) += point(1) * point(1);
        cumulants(7) += point(1) * point(2);
        cumulants(8) += point(2) * point(2);
    }

    cumulants /= (double)input.points_.size();

    mean(0) = cumulants(0);
    mean(1) = cumulants(1);
    mean(2) = cumulants(2);

    covariance(0, 0) = cumulants(3) - cumulants(0) * cumulants(0);
    covariance(1, 1) = cumulants(6) - cumulants(1) * cumulants(1);
    covariance(2, 2) = cumulants(8) - cumulants(2) * cumulants(2);
    covariance(0, 1) = cumulants(4) - cumulants(0) * cumulants(1);
    covariance(1, 0) = covariance(0, 1);
    covariance(0, 2) = cumulants(5) - cumulants(0) * cumulants(2);
    covariance(2, 0) = covariance(0, 2);
    covariance(1, 2) = cumulants(7) - cumulants(1) * cumulants(2);
    covariance(2, 1) = covariance(1, 2);

    return std::make_tuple(mean, covariance);
}

std::vector<double> ComputePointCloudMahalanobisDistance(PointCloud &input) {
    std::vector<double> mahalanobis(input.points_.size());
    Eigen::Vector3d mean;
    Eigen::Matrix3d covariance;
    std::tie(mean, covariance) = ComputePointCloudMeanAndCovariance(input);
    Eigen::Matrix3d cov_inv = covariance.inverse();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < (int)input.points_.size(); i++) {
        Eigen::Vector3d p = input.points_[i] - mean;
        mahalanobis[i] = std::sqrt(p.transpose() * cov_inv * p);
    }
    return mahalanobis;
}

std::vector<double> ComputePointCloudNearestNeighborDistance(
        const PointCloud &input) {
    std::vector<double> nn_dis(input.points_.size());
    KDTreeFlann kdtree(input);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < (int)input.points_.size(); i++) {
        std::vector<int> indices(2);
        std::vector<double> dists(2);
        if (kdtree.SearchKNN(input.points_[i], 2, indices, dists) <= 1) {
            utility::PrintDebug(
                    "[ComputePointCloudNearestNeighborDistance] Found a point "
                    "without neighbors.\n");
            nn_dis[i] = 0.0;
        } else {
            nn_dis[i] = std::sqrt(dists[1]);
        }
    }
    return nn_dis;
}

#ifdef OPEN3D_USE_CUDA

std::tuple<Eigen::Vector3d, Eigen::Matrix3d>
ComputePointCloudMeanAndCovarianceGPU(PointCloud &input) {
    auto default_output = std::make_tuple(Eigen::Vector3d::Zero(),
                                          Eigen::Matrix3d::Identity());

    size_t nr_points = input.points_.size();
    double *d_points = input.points_.d_data;
    cuda::DeviceID::Type device_id = input.points_.device_id;

    cudaError_t status = cudaSuccess;

    // host memory
    vector<Eigen::Matrix3d> h_cumulants(nr_points);

    int outputSize = h_cumulants.size() * 9;  // Mat3d::Size;

    // allocate temporary device memory
    double *d_cumulants = NULL;
    status = cuda::AllocateDeviceMemory(&d_cumulants, outputSize, device_id);
    cuda::DebugInfo("ComputePointCloudMeanAndCovarianceGPU:01", status);
    if (cudaSuccess != status) return default_output;

    // execute on GPU
    status = meanAndCovarianceAccumulatorHelper(device_id, d_points, nr_points,
                                                d_cumulants);
    cuda::DebugInfo("ComputePointCloudMeanAndCovarianceGPU:02", status);
    if (cudaSuccess != status) return default_output;

    // Copy results to the host
    status = cuda::CopyDev2HstMemory(d_cumulants, (double *)&h_cumulants[0],
                                     outputSize);
    cuda::DebugInfo("ComputePointCloudMeanAndCovarianceGPU:03", status);
    if (cudaSuccess != status) return default_output;

    // Free temporary device memory
    status = cuda::ReleaseDeviceMemory(&d_cumulants);
    cuda::DebugInfo("ComputePointCloudMeanAndCovarianceGPU:04", status);
    if (cudaSuccess != status) return default_output;

    // initialize with zeros
    Eigen::Matrix3d cumulant = Eigen::Matrix3d::Zero();
    for (int i = 0; i < h_cumulants.size(); i++) cumulant += h_cumulants[i];

    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    mean(0) = cumulant(0);
    mean(1) = cumulant(1);
    mean(2) = cumulant(2);

    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    covariance(0, 0) = cumulant(3) - cumulant(0) * cumulant(0);
    covariance(1, 1) = cumulant(6) - cumulant(1) * cumulant(1);
    covariance(2, 2) = cumulant(8) - cumulant(2) * cumulant(2);
    covariance(0, 1) = cumulant(4) - cumulant(0) * cumulant(1);
    covariance(1, 0) = covariance(0, 1);
    covariance(0, 2) = cumulant(5) - cumulant(0) * cumulant(2);
    covariance(2, 0) = covariance(0, 2);
    covariance(1, 2) = cumulant(7) - cumulant(1) * cumulant(2);
    covariance(2, 1) = covariance(1, 2);

    return std::make_tuple(mean, covariance);
}

#endif  // OPEN3D_USE_CUDA

}  // namespace geometry
}  // namespace open3d
