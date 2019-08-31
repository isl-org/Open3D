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

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "Open3D/Geometry/Geometry.h"
#include "Open3D/Utility/Eigen.h"

namespace open3d {
namespace geometry {

class AxisAlignedBoundingBox;
class OrientedBoundingBox;

class Geometry3D : public Geometry {
public:
    enum class RotationType { XYZ, YZX, ZXY, XZY, ZYX, YXZ, AxisAngle };

    ~Geometry3D() override {}

protected:
    Geometry3D(GeometryType type) : Geometry(type, 3) {}

public:
    Geometry3D& Clear() override = 0;
    bool IsEmpty() const override = 0;
    virtual Eigen::Vector3d GetMinBound() const = 0;
    virtual Eigen::Vector3d GetMaxBound() const = 0;
    virtual Eigen::Vector3d GetCenter() const = 0;
    virtual AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const = 0;
    virtual OrientedBoundingBox GetOrientedBoundingBox() const = 0;
    virtual Geometry3D& Transform(const Eigen::Matrix4d& transformation) = 0;
    virtual Geometry3D& Translate(const Eigen::Vector3d& translation,
                                  bool relative = true) = 0;
    virtual Geometry3D& Scale(const double scale, bool center = true) = 0;
    virtual Geometry3D& Rotate(const Eigen::Vector3d& rotation,
                               bool center = true,
                               RotationType type = RotationType::XYZ) = 0;

protected:
    Eigen::Vector3d ComputeMinBound(
            const std::vector<Eigen::Vector3d>& points) const;
    Eigen::Vector3d ComputeMaxBound(
            const std::vector<Eigen::Vector3d>& points) const;
    Eigen::Vector3d ComputeCenter(
            const std::vector<Eigen::Vector3d>& points) const;

    void ResizeAndPaintUniformColor(std::vector<Eigen::Vector3d>& colors,
                                    const size_t size,
                                    const Eigen::Vector3d& color) const;

    void TransformPoints(const Eigen::Matrix4d& transformation,
                         std::vector<Eigen::Vector3d>& points) const;
    void TransformNormals(const Eigen::Matrix4d& transformation,
                          std::vector<Eigen::Vector3d>& normals) const;
    void TranslatePoints(const Eigen::Vector3d& translation,
                         std::vector<Eigen::Vector3d>& points,
                         bool relative) const;
    void ScalePoints(const double scale,
                     std::vector<Eigen::Vector3d>& points,
                     bool center) const;
    void RotatePoints(const Eigen::Vector3d& rotation,
                      std::vector<Eigen::Vector3d>& points,
                      bool center,
                      RotationType type) const;
    void RotateNormals(const Eigen::Vector3d& rotation,
                       std::vector<Eigen::Vector3d>& normals,
                       bool center,
                       RotationType type) const;

    Eigen::Matrix3d GetRotationMatrix(
            const Eigen::Vector3d& rotation,
            RotationType type = RotationType::XYZ) const;
};

}  // namespace geometry
}  // namespace open3d
