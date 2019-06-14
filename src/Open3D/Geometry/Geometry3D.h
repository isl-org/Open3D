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
    virtual Geometry3D& Transform(const Eigen::Matrix4d& transformation) = 0;
    virtual Geometry3D& Translate(const Eigen::Vector3d& translation) = 0;
    virtual Geometry3D& Scale(const double scale, bool center = true) = 0;
    virtual Geometry3D& Rotate(const Eigen::Vector3d& rotation,
                               bool center = true,
                               RotationType type = RotationType::XYZ) = 0;

protected:
    Eigen::Matrix3d GetRotationMatrix(
            const Eigen::Vector3d& rotation,
            RotationType type = RotationType::XYZ) const {
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
};

}  // namespace geometry
}  // namespace open3d
