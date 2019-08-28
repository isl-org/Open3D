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
          x_axis_(0, 0, 0),
          y_axis_(0, 0, 0),
          z_axis_(0, 0, 0),
          color_(0, 0, 0) {}
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
    virtual OrientedBoundingBox& Rotate(
            const Eigen::Vector3d& rotation,
            bool center = true,
            RotationType type = RotationType::XYZ) override;

    double Volume() const;
    std::vector<Eigen::Vector3d> GetBoxPoints() const;

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
    Eigen::Vector3d x_axis_;
    Eigen::Vector3d y_axis_;
    Eigen::Vector3d z_axis_;
    Eigen::Vector3d color_;
};

class AxisAlignedBoundingBox : public Geometry3D {
public:
    AxisAlignedBoundingBox()
        : Geometry3D(Geometry::GeometryType::AxisAlignedBoundingBox),
          min_bound_(0, 0, 0),
          max_bound_(0, 0, 0),
          color_(0, 0, 0) {}
    ~AxisAlignedBoundingBox() override {}

public:
    AxisAlignedBoundingBox& Clear() override;
    bool IsEmpty() const override;
    virtual Eigen::Vector3d GetMinBound() const override;
    virtual Eigen::Vector3d GetMaxBound() const override;
    virtual Eigen::Vector3d GetCenter() const override;
    virtual AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const override;
    virtual OrientedBoundingBox GetOrientedBoundingBox() const override;
    virtual AxisAlignedBoundingBox& Transform(
            const Eigen::Matrix4d& transformation) override;
    virtual AxisAlignedBoundingBox& Translate(
            const Eigen::Vector3d& translation, bool relative = true) override;
    virtual AxisAlignedBoundingBox& Scale(const double scale,
                                          bool center = true) override;
    virtual AxisAlignedBoundingBox& Rotate(
            const Eigen::Vector3d& rotation,
            bool center = true,
            RotationType type = RotationType::XYZ) override;

    AxisAlignedBoundingBox& operator+=(const AxisAlignedBoundingBox& other) {
        min_bound_ = min_bound_.array().min(other.min_bound_.array()).matrix();
        max_bound_ = max_bound_.array().max(other.max_bound_.array()).matrix();
        return *this;
    }

    double Volume() const;
    std::vector<Eigen::Vector3d> GetBoxPoints() const;

    Eigen::Vector3d GetExtend() const { return (max_bound_ - min_bound_); }

    Eigen::Vector3d GetHalfExtend() const { return GetExtend() * 0.5; }

    double GetMaxExtend() const { return (max_bound_ - min_bound_).maxCoeff(); }

    double GetXPercentage(double x) const {
        return (x - min_bound_(0)) / (max_bound_(0) - min_bound_(0));
    }

    double GetYPercentage(double y) const {
        return (y - min_bound_(1)) / (max_bound_(1) - min_bound_(1));
    }

    double GetZPercentage(double z) const {
        return (z - min_bound_(2)) / (max_bound_(2) - min_bound_(2));
    }

    std::string GetPrintInfo() const;

    static AxisAlignedBoundingBox CreateFromPoints(
            const std::vector<Eigen::Vector3d>& points);

public:
    Eigen::Vector3d min_bound_;
    Eigen::Vector3d max_bound_;
    Eigen::Vector3d color_;
};

}  // namespace geometry
}  // namespace open3d
