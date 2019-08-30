// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 www.open3d.org
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

#include "Open3D/Geometry/Geometry3D.h"

#include <Eigen/Dense>
#include <numeric>

#include "Open3D/Utility/Console.h"

namespace open3d {
namespace geometry {

Eigen::Vector3d Geometry3D::ComputeMinBound(
        const std::vector<Eigen::Vector3d>& points) const {
    if (points.empty()) {
        return Eigen::Vector3d(0.0, 0.0, 0.0);
    }
    return std::accumulate(
            points.begin(), points.end(), points[0],
            [](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
                return a.array().min(b.array()).matrix();
            });
}

Eigen::Vector3d Geometry3D::ComputeMaxBound(
        const std::vector<Eigen::Vector3d>& points) const {
    if (points.empty()) {
        return Eigen::Vector3d(0.0, 0.0, 0.0);
    }
    return std::accumulate(
            points.begin(), points.end(), points[0],
            [](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
                return a.array().max(b.array()).matrix();
            });
}
Eigen::Vector3d Geometry3D::ComputeCenter(
        const std::vector<Eigen::Vector3d>& points) const {
    Eigen::Vector3d center(0, 0, 0);
    if (points.empty()) {
        return center;
    }
    center = std::accumulate(points.begin(), points.end(), center);
    center /= double(points.size());
    return center;
}

void Geometry3D::ResizeAndPaintUniformColor(
        std::vector<Eigen::Vector3d>& colors,
        const size_t size,
        const Eigen::Vector3d& color) const {
    colors.resize(size);
    Eigen::Vector3d clipped_color = color;
    if (color.minCoeff() < 0 || color.maxCoeff() > 1) {
        utility::LogWarning(
                "invalid color in PaintUniformColor, clipping to [0, 1]\n");
        clipped_color = clipped_color.array()
                                .max(Eigen::Vector3d(0, 0, 0).array())
                                .matrix();
        clipped_color = clipped_color.array()
                                .min(Eigen::Vector3d(1, 1, 1).array())
                                .matrix();
    }
    for (size_t i = 0; i < size; i++) {
        colors[i] = clipped_color;
    }
}

void Geometry3D::TransformPoints(const Eigen::Matrix4d& transformation,
                                 std::vector<Eigen::Vector3d>& points) const {
    for (auto& point : points) {
        Eigen::Vector4d new_point =
                transformation *
                Eigen::Vector4d(point(0), point(1), point(2), 1.0);
        point = new_point.head<3>() / new_point(3);
    }
}

void Geometry3D::TransformNormals(const Eigen::Matrix4d& transformation,
                                  std::vector<Eigen::Vector3d>& normals) const {
    for (auto& normal : normals) {
        Eigen::Vector4d new_normal =
                transformation *
                Eigen::Vector4d(normal(0), normal(1), normal(2), 0.0);
        normal = new_normal.head<3>();
    }
}

void Geometry3D::TranslatePoints(const Eigen::Vector3d& translation,
                                 std::vector<Eigen::Vector3d>& points,
                                 bool relative) const {
    Eigen::Vector3d transform = translation;
    if (!relative) {
        transform -= ComputeCenter(points);
    }
    for (auto& point : points) {
        point += transform;
    }
}

void Geometry3D::ScalePoints(const double scale,
                             std::vector<Eigen::Vector3d>& points,
                             bool center) const {
    Eigen::Vector3d points_center(0, 0, 0);
    if (center && !points.empty()) {
        points_center = ComputeCenter(points);
    }
    for (auto& point : points) {
        point = (point - points_center) * scale + points_center;
    }
}

void Geometry3D::RotatePoints(const Eigen::Vector3d& rotation,
                              std::vector<Eigen::Vector3d>& points,
                              bool center,
                              RotationType type) const {
    Eigen::Vector3d points_center(0, 0, 0);
    if (center && !points.empty()) {
        points_center = ComputeCenter(points);
    }
    const Eigen::Matrix3d R = GetRotationMatrix(rotation, type);
    for (auto& point : points) {
        point = R * (point - points_center) + points_center;
    }
}

void Geometry3D::RotateNormals(const Eigen::Vector3d& rotation,
                               std::vector<Eigen::Vector3d>& normals,
                               bool center,
                               RotationType type) const {
    const Eigen::Matrix3d R = GetRotationMatrix(rotation, type);
    for (auto& normal : normals) {
        normal = R * normal;
    }
}

Eigen::Matrix3d Geometry3D::GetRotationMatrix(const Eigen::Vector3d& rotation,
                                              RotationType type) const {
    if (type == RotationType::XYZ) {
        return open3d::utility::RotationMatrixX(rotation(0)) *
               open3d::utility::RotationMatrixY(rotation(1)) *
               open3d::utility::RotationMatrixZ(rotation(2));
    } else if (type == RotationType::YZX) {
        return open3d::utility::RotationMatrixY(rotation(0)) *
               open3d::utility::RotationMatrixZ(rotation(1)) *
               open3d::utility::RotationMatrixX(rotation(2));
    } else if (type == RotationType::ZXY) {
        return open3d::utility::RotationMatrixZ(rotation(0)) *
               open3d::utility::RotationMatrixX(rotation(1)) *
               open3d::utility::RotationMatrixY(rotation(2));
    } else if (type == RotationType::XZY) {
        return open3d::utility::RotationMatrixX(rotation(0)) *
               open3d::utility::RotationMatrixZ(rotation(1)) *
               open3d::utility::RotationMatrixY(rotation(2));
    } else if (type == RotationType::ZYX) {
        return open3d::utility::RotationMatrixZ(rotation(0)) *
               open3d::utility::RotationMatrixY(rotation(1)) *
               open3d::utility::RotationMatrixX(rotation(2));
    } else if (type == RotationType::YXZ) {
        return open3d::utility::RotationMatrixY(rotation(0)) *
               open3d::utility::RotationMatrixX(rotation(1)) *
               open3d::utility::RotationMatrixZ(rotation(2));
    } else if (type == RotationType::AxisAngle) {
        const double phi = rotation.norm();
        return Eigen::AngleAxisd(phi, rotation / phi).toRotationMatrix();
    } else {
        return Eigen::Matrix3d::Identity();
    }
}

}  // namespace geometry
}  // namespace open3d
