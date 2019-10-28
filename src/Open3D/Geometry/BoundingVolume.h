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

#include "Open3D/Geometry/Geometry3D.h"

namespace open3d {
namespace geometry {

class AxisAlignedBoundingBox;

class OrientedBoundingBox : public Geometry3D {
public:
    OrientedBoundingBox()
        : Geometry3D(Geometry::GeometryType::OrientedBoundingBox),
          center_(0, 0, 0),
          R_(Eigen::Matrix3d::Identity()),
          extent_(0, 0, 0),
          color_(0, 0, 0) {}
    OrientedBoundingBox(const Eigen::Vector3d& center,
                        const Eigen::Matrix3d& R,
                        const Eigen::Vector3d& extent)
        : Geometry3D(Geometry::GeometryType::OrientedBoundingBox),
          center_(center),
          R_(R),
          extent_(extent) {}
    ~OrientedBoundingBox() override {}

public:
    OrientedBoundingBox& Clear() override;
    bool IsEmpty() const override;
    virtual Eigen::Vector3d GetMinBound() const override;
    virtual Eigen::Vector3d GetMaxBound() const override;
    virtual Eigen::Vector3d GetCenter() const override;
    virtual AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const override;
    virtual OrientedBoundingBox GetOrientedBoundingBox() const override;
    virtual OrientedBoundingBox& Transform(
            const Eigen::Matrix4d& transformation) override;
    virtual OrientedBoundingBox& Translate(const Eigen::Vector3d& translation,
                                           bool relative = true) override;
    virtual OrientedBoundingBox& Scale(const double scale,
                                       bool center = true) override;
    virtual OrientedBoundingBox& Rotate(const Eigen::Matrix3d& R,
                                        bool center = true) override;

    double Volume() const;
    std::vector<Eigen::Vector3d> GetBoxPoints() const;

    /// Return indices to points that are within the bounding box.
    std::vector<size_t> GetPointIndicesWithinBoundingBox(
            const std::vector<Eigen::Vector3d>& points) const;

    static OrientedBoundingBox CreateFromAxisAlignedBoundingBox(
            const AxisAlignedBoundingBox& aabox);

    /// Creates an oriented bounding box using a PCA.
    /// Note, that this is only an approximation to the minimum oriented
    /// bounding box that could be computed for example with O'Rourke's
    /// algorithm (cf. http://cs.smith.edu/~jorourke/Papers/MinVolBox.pdf,
    /// https://www.geometrictools.com/Documentation/MinimumVolumeBox.pdf)
    static OrientedBoundingBox CreateFromPoints(
            const std::vector<Eigen::Vector3d>& points);

public:
    Eigen::Vector3d center_;
    Eigen::Matrix3d R_;
    Eigen::Vector3d extent_;
    Eigen::Vector3d color_;
};

/// Class that defines an axis_aligned box that can be computed from 3D geometries. The axis aligned bounding box uses the cooridnate axes for bounding box generation
class AxisAlignedBoundingBox : public Geometry3D {
public:
    AxisAlignedBoundingBox()
        : Geometry3D(Geometry::GeometryType::AxisAlignedBoundingBox),
          min_bound_(0, 0, 0),
          max_bound_(0, 0, 0),
          color_(0, 0, 0) {}
    AxisAlignedBoundingBox(const Eigen::Vector3d& min_bound,
                           const Eigen::Vector3d& max_bound)
        : Geometry3D(Geometry::GeometryType::AxisAlignedBoundingBox),
          min_bound_(min_bound),
          max_bound_(max_bound),
          color_(0, 0, 0) {}
    ~AxisAlignedBoundingBox() override {}

public:
    /// Clear all elements in the geometry
    AxisAlignedBoundingBox& Clear() override;
    /// Returns True iff the geometry is empty
    bool IsEmpty() const override;
    /// Returns min bounds for geometry coordinates
    virtual Eigen::Vector3d GetMinBound() const override;
    /// Returns max bounds for geometry coordinates
    virtual Eigen::Vector3d GetMaxBound() const override;
    /// Returns the center of the geometry coordinates
    virtual Eigen::Vector3d GetCenter() const override;
    /// Returns an axis-aligned bounding box of the geometry
    virtual AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const override;
    /// Returns an oriented bounding box of the geometry
    virtual OrientedBoundingBox GetOrientedBoundingBox() const override;
    /// Apply transformation (4x4 matrix) to the geometry coordinates
    virtual AxisAlignedBoundingBox& Transform(
            const Eigen::Matrix4d& transformation) override;
    /// Apply translation to the geometry coordinates.
    /// Parameters:
    /// translation - A 3D vector to transform the geometry
    /// relative - If true, the translation vector is directly added to the geometry coordinates. Otherwise, the center is moved to the translation vector.
    virtual AxisAlignedBoundingBox& Translate(
            const Eigen::Vector3d& translation, bool relative = true) override;
    /// Apply scaling to the geometry coordinates
    /// Parameters:
    /// scale - The scale parameter that is multiplied to the points/vertices of the geometry
    /// center - If true, then the scale is applied to the centered geometry
    virtual AxisAlignedBoundingBox& Scale(const double scale,
                                          bool center = true) override;
    /// Apply rotation to the geometry coordinates and normals
    /// Parameters:
    /// rotation -  A 3D vector that either defines the three angles for Euler rotation, or in the axis-angle representation the normalized vector defines the axis of rotation and the norm the angle around this axis
    /// center -  If true, then the rotation is applied to the centered geometry
    /// type - Type of rotation, i.e., an Euler format, or axis-angle.
    virtual AxisAlignedBoundingBox& Rotate(const Eigen::Matrix3d& R,
                                           bool center = true) override;

    AxisAlignedBoundingBox& operator+=(const AxisAlignedBoundingBox& other);

    Eigen::Vector3d GetExtent() const { return (max_bound_ - min_bound_); }

    Eigen::Vector3d GetHalfExtent() const { return GetExtent() * 0.5; }

    double GetMaxExtent() const { return (max_bound_ - min_bound_).maxCoeff(); }

    double GetXPercentage(double x) const {
        return (x - min_bound_(0)) / (max_bound_(0) - min_bound_(0));
    }

    double GetYPercentage(double y) const {
        return (y - min_bound_(1)) / (max_bound_(1) - min_bound_(1));
    }

    double GetZPercentage(double z) const {
        return (z - min_bound_(2)) / (max_bound_(2) - min_bound_(2));
    }

    /// Returns the volume of the bounding box
    double Volume() const;
    /// Returns the eight points that define the bounding box
    std::vector<Eigen::Vector3d> GetBoxPoints() const;

    /// Return indices to points that are within the bounding box.
    std::vector<size_t> GetPointIndicesWithinBoundingBox(
            const std::vector<Eigen::Vector3d>& points) const;

    /// Returns the 3D dimensions of the bounding box in string format
    std::string GetPrintInfo() const;

    /// Creates the bounding box that encloses the set of points
    static AxisAlignedBoundingBox CreateFromPoints(
            const std::vector<Eigen::Vector3d>& points);

public:
    /// The lower x, y, z bounds of the bounding box
    Eigen::Vector3d min_bound_;
    /// The upper x, y, z bounds of the bounding box
    Eigen::Vector3d max_bound_;
    Eigen::Vector3d color_;
};

}  // namespace geometry
}  // namespace open3d
