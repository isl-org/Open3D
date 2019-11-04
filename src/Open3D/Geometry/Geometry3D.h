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

/// \class Geometry3D
///
/// \brief The base geometry class for 3D geometries.
///
/// Main class for 3D geometries, Derives all data from Geometry Base class.
class Geometry3D : public Geometry {
public:
    ~Geometry3D() override {}

protected:
    /// \brief Parameterized Constructor.
    ///
    /// \param type - type of object based on GeometryType.
    Geometry3D(GeometryType type) : Geometry(type, 3) {}

public:
    /// Clear all elements in the geometry.
    Geometry3D& Clear() override = 0;
    /// Returns `true` iff the geometry is empty.
    bool IsEmpty() const override = 0;
    /// Returns min bounds for geometry coordinates.
    virtual Eigen::Vector3d GetMinBound() const = 0;
    /// Returns max bounds for geometry coordinates.
    virtual Eigen::Vector3d GetMaxBound() const = 0;
    /// Returns the center of the geometry coordinates.
    virtual Eigen::Vector3d GetCenter() const = 0;
    /// Returns an axis-aligned bounding box of the geometry.
    virtual AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const = 0;
    /// Returns an oriented bounding box of the geometry.
    virtual OrientedBoundingBox GetOrientedBoundingBox() const = 0;
    /// \brief Apply transformation (4x4 matrix) to the geometry coordinates.
    virtual Geometry3D& Transform(const Eigen::Matrix4d& transformation) = 0;
    /// \brief Apply translation to the geometry coordinates.
    ///
    /// \param translation - A 3D vector to transform the geometry.
    /// \param relative - If `true`, the translation vector is directly added to
    /// the geometry coordinates. Otherwise, the center is moved to the
    /// translation vector.
    virtual Geometry3D& Translate(const Eigen::Vector3d& translation,
                                  bool relative = true) = 0;
    /// \brief Apply scaling to the geometry coordinates.
    ///
    /// \param scale -  The scale parameter that is multiplied to the
    /// points/vertices of the geometry.
    /// \param center - If `true`, then the scale is applied to the centered
    /// geometry.
    virtual Geometry3D& Scale(const double scale, bool center = true) = 0;
    /// \brief Apply rotation to the geometry coordinates and normals.
    ///
    /// \param R - A 3D vector that either defines the three angles for Euler
    /// rotation, or in the axis-angle representation the normalized vector
    /// defines the axis of rotation and the norm the angle around this axis.
    /// \param center - If `true`, then the rotation is applied to the centered
    /// geometry.
    virtual Geometry3D& Rotate(const Eigen::Matrix3d& R,
                               bool center = true) = 0;

    /// Get Rotation Matrix from XYZ RotationType.
    static Eigen::Matrix3d GetRotationMatrixFromXYZ(
            const Eigen::Vector3d& rotation);
    /// Get Rotation Matrix from YZX RotationType.
    static Eigen::Matrix3d GetRotationMatrixFromYZX(
            const Eigen::Vector3d& rotation);
    /// Get Rotation Matrix from ZXY RotationType.
    static Eigen::Matrix3d GetRotationMatrixFromZXY(
            const Eigen::Vector3d& rotation);
    /// Get Rotation Matrix from XZY RotationType.
    static Eigen::Matrix3d GetRotationMatrixFromXZY(
            const Eigen::Vector3d& rotation);
    /// Get Rotation Matrix from ZYX RotationType.
    static Eigen::Matrix3d GetRotationMatrixFromZYX(
            const Eigen::Vector3d& rotation);
    /// Get Rotation Matrix from YXZ RotationType.
    static Eigen::Matrix3d GetRotationMatrixFromYXZ(
            const Eigen::Vector3d& rotation);
    /// Get Rotation Matrix from AxisAngle RotationType.
    static Eigen::Matrix3d GetRotationMatrixFromAxisAngle(
            const Eigen::Vector3d& rotation);
    /// Get Rotation Matrix from Quaternion.
    static Eigen::Matrix3d GetRotationMatrixFromQuaternion(
            const Eigen::Vector4d& rotation);

protected:
    /// Compute min bound of a set points.
    Eigen::Vector3d ComputeMinBound(
            const std::vector<Eigen::Vector3d>& points) const;
    /// Compute max bound of a set points.
    Eigen::Vector3d ComputeMaxBound(
            const std::vector<Eigen::Vector3d>& points) const;
    /// Computer center of a set of points.
    Eigen::Vector3d ComputeCenter(
            const std::vector<Eigen::Vector3d>& points) const;

    /// \fn ResizeAndPaintUniformColor
    ///
    /// \brief Resizes the colors vector and paints a uniform color.
    ///
    /// \param colors - An array of eigen vectors specifies colors in RGB.
    /// \param size - The resultant size of the colors array.
    /// \param color - The final color in which the colors will be painted.
    void ResizeAndPaintUniformColor(std::vector<Eigen::Vector3d>& colors,
                                    const size_t size,
                                    const Eigen::Vector3d& color) const;

    /// Transforms all points along the transformation matrix.
    void TransformPoints(const Eigen::Matrix4d& transformation,
                         std::vector<Eigen::Vector3d>& points) const;
    /// Transforms all normals along the transformation matrix.
    void TransformNormals(const Eigen::Matrix4d& transformation,
                          std::vector<Eigen::Vector3d>& normals) const;
    /// Translate all points along the translation matrix.
    void TranslatePoints(const Eigen::Vector3d& translation,
                         std::vector<Eigen::Vector3d>& points,
                         bool relative) const;
    /// Scale all points by the scaling factor.
    void ScalePoints(const double scale,
                     std::vector<Eigen::Vector3d>& points,
                     bool center) const;
    /// Rotate all points along the rotation matrix.
    void RotatePoints(const Eigen::Matrix3d& R,
                      std::vector<Eigen::Vector3d>& points,
                      bool center) const;
    /// Rotate all normals along the rotation matrix.
    void RotateNormals(const Eigen::Matrix3d& R,
                       std::vector<Eigen::Vector3d>& normals,
                       bool center) const;
};

}  // namespace geometry
}  // namespace open3d
