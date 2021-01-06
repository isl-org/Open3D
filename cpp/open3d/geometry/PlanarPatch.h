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
#include <memory>
#include <vector>

#include "open3d/geometry/Geometry3D.h"
#include "open3d/geometry/PointCloud.h"

namespace open3d {
namespace geometry {

/// \class PlanarPatch
///
/// \brief A planar patch in 3D, typically detected from a point cloud.
class PlanarPatch : public Geometry3D {
public:
    /// \brief Default Constructor.
    PlanarPatch() : Geometry3D(Geometry::GeometryType::PlanarPatch) {}
    ~PlanarPatch() override {}

public:
    PlanarPatch& Clear() override;
    bool IsEmpty() const override;
    Eigen::Vector3d GetMinBound() const override;
    Eigen::Vector3d GetMaxBound() const override;
    Eigen::Vector3d GetCenter() const override;
    AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const override;
    OrientedBoundingBox GetOrientedBoundingBox() const override;
    PlanarPatch& Transform(const Eigen::Matrix4d& transformation) override;
    PlanarPatch& Translate(const Eigen::Vector3d& translation,
                           bool relative = true) override;
    PlanarPatch& Scale(const double scale,
                       const Eigen::Vector3d& center) override;
    PlanarPatch& Rotate(const Eigen::Matrix3d& R,
                        const Eigen::Vector3d& center) override;

    /// \brief Assigns a color to the PlanarPatch.
    ///
    /// \param color  RGB colors of points.
    PlanarPatch &PaintUniformColor(const Eigen::Vector3d &color);

    /// \brief Compute the (signed) distance from the planar surface to a point.
    ///
    /// \param point  Point to find distance to.
    double GetSignedDistanceToPoint(const Eigen::Vector3d& point) const;

public:
    Eigen::Vector3d center_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal_ = Eigen::Vector3d::Zero();
    /// "d" parameter of a plane represented as ax + by + cz + d = 0
    double dist_from_origin_ = 0;
    Eigen::Vector3d basis_x_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d basis_y_ = Eigen::Vector3d::Zero();

    Eigen::Vector3d color_ = Eigen::Vector3d::Zero();

};

}  // namespace geometry
}  // namespace open3d
