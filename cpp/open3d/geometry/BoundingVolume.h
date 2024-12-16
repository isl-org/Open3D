// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>

#include "open3d/geometry/Geometry3D.h"

namespace open3d {
namespace geometry {

class AxisAlignedBoundingBox;

/// \class OrientedBoundingBox
///
/// \brief A bounding box oriented along an arbitrary frame of reference.
///
/// The oriented bounding box is defined by its center position, rotation
/// matrix and extent.
class OrientedBoundingBox : public Geometry3D {
public:
    /// \brief Default constructor.
    ///
    /// Creates an empty Oriented Bounding Box.
    OrientedBoundingBox()
        : Geometry3D(Geometry::GeometryType::OrientedBoundingBox),
          center_(0, 0, 0),
          R_(Eigen::Matrix3d::Identity()),
          extent_(0, 0, 0),
          color_(1, 1, 1) {}
    /// \brief Parameterized constructor.
    ///
    /// \param center Specifies the center position of the bounding box.
    /// \param R The rotation matrix specifying the orientation of the
    /// bounding box with the original frame of reference.
    /// \param extent The extent of the bounding box.
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

    /// Creates an axis-aligned bounding box around the
    /// points (corners) of the object.
    virtual AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const override;

    /// Returns the object itself
    virtual OrientedBoundingBox GetOrientedBoundingBox(
            bool robust) const override;

    /// Returns the object itself
    virtual OrientedBoundingBox GetMinimalOrientedBoundingBox(
            bool robust) const override;

    virtual OrientedBoundingBox& Transform(
            const Eigen::Matrix4d& transformation) override;
    virtual OrientedBoundingBox& Translate(const Eigen::Vector3d& translation,
                                           bool relative = true) override;
    virtual OrientedBoundingBox& Scale(const double scale,
                                       const Eigen::Vector3d& center) override;
    virtual OrientedBoundingBox& Rotate(const Eigen::Matrix3d& R,
                                        const Eigen::Vector3d& center) override;

    /// Returns the volume of the bounding box.
    double Volume() const;

    /// Returns the eight points that define the bounding box.
    /// \verbatim
    ///      ------- x
    ///     /|
    ///    / |
    ///   /  | z
    ///  y
    ///      0 ------------------- 1
    ///       /|                /|
    ///      / |               / |
    ///     /  |              /  |
    ///    /   |             /   |
    /// 2 ------------------- 7  |
    ///   |    |____________|____| 6
    ///   |   /3            |   /
    ///   |  /              |  /
    ///   | /               | /
    ///   |/                |/
    /// 5 ------------------- 4
    /// \endverbatim
    std::vector<Eigen::Vector3d> GetBoxPoints() const;

    /// Return indices to points that are within the bounding box.
    std::vector<size_t> GetPointIndicesWithinBoundingBox(
            const std::vector<Eigen::Vector3d>& points) const;

    /// Returns an oriented bounding box from the AxisAlignedBoundingBox.
    ///
    /// \param aabox AxisAlignedBoundingBox object from which
    /// OrientedBoundingBox is created.
    static OrientedBoundingBox CreateFromAxisAlignedBoundingBox(
            const AxisAlignedBoundingBox& aabox);

    /// Creates an oriented bounding box using a PCA.
    /// Note, that this is only an approximation to the minimum oriented
    /// bounding box that could be computed for example with O'Rourke's
    /// algorithm (cf. http://cs.smith.edu/~jorourke/Papers/MinVolBox.pdf,
    /// https://www.geometrictools.com/Documentation/MinimumVolumeBox.pdf)
    /// \param points The input points
    /// \param robust If set to true uses a more robust method which works
    ///               in degenerate cases but introduces noise to the points
    ///               coordinates.
    static OrientedBoundingBox CreateFromPoints(
            const std::vector<Eigen::Vector3d>& points, bool robust = false);

    /// Creates the oriented bounding box with the smallest volume.
    /// The algorithm makes use of the fact that at least one edge of
    /// the convex hull must be collinear with an edge of the minimum
    /// bounding box: for each triangle in the convex hull, calculate
    /// the minimal axis aligned box in the frame of that triangle.
    /// at the end, return the box with the smallest volume
    /// \param points The input points
    /// \param robust If set to true uses a more robust method which works
    ///               in degenerate cases but introduces noise to the points
    ///               coordinates.
    static OrientedBoundingBox CreateFromPointsMinimal(
            const std::vector<Eigen::Vector3d>& points, bool robust = false);

public:
    /// The center point of the bounding box.
    Eigen::Vector3d center_;
    /// The rotation matrix of the bounding box to transform the original frame
    /// of reference to the frame of this box.
    Eigen::Matrix3d R_;
    /// The extent of the bounding box in its frame of reference.
    Eigen::Vector3d extent_;
    /// The color of the bounding box in RGB.
    Eigen::Vector3d color_;
};

/// \class AxisAlignedBoundingBox
///
/// \brief A bounding box that is aligned along the coordinate axes and defined
/// by the min_bound and max_bound.
///
///  The AxisAlignedBoundingBox uses the coordinate axes for bounding box
///  generation. This means that the bounding box is oriented along the
///  coordinate axes.
class AxisAlignedBoundingBox : public Geometry3D {
public:
    /// \brief Default constructor.
    ///
    /// Creates an empty Axis Aligned Bounding Box.
    AxisAlignedBoundingBox()
        : Geometry3D(Geometry::GeometryType::AxisAlignedBoundingBox),
          min_bound_(0, 0, 0),
          max_bound_(0, 0, 0),
          color_(1, 1, 1) {}
    /// \brief Parameterized constructor.
    ///
    /// \param min_bound Lower bounds of the bounding box for all axes.
    /// \param max_bound Upper bounds of the bounding box for all axes.
    AxisAlignedBoundingBox(const Eigen::Vector3d& min_bound,
                           const Eigen::Vector3d& max_bound);
    ~AxisAlignedBoundingBox() override {}

public:
    AxisAlignedBoundingBox& Clear() override;
    bool IsEmpty() const override;
    virtual Eigen::Vector3d GetMinBound() const override;
    virtual Eigen::Vector3d GetMaxBound() const override;
    virtual Eigen::Vector3d GetCenter() const override;

    /// Returns the object itself
    virtual AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const override;

    /// Returns an oriented bounding box with the same dimensions
    /// and orientation as the object
    virtual OrientedBoundingBox GetOrientedBoundingBox(
            bool robust = false) const override;

    /// Returns an oriented bounding box with the same dimensions
    /// and orientation as the object
    virtual OrientedBoundingBox GetMinimalOrientedBoundingBox(
            bool robust = false) const override;
    virtual AxisAlignedBoundingBox& Transform(
            const Eigen::Matrix4d& transformation) override;
    virtual AxisAlignedBoundingBox& Translate(
            const Eigen::Vector3d& translation, bool relative = true) override;

    /// \brief Scales the axis-aligned bounding boxes.
    /// If \f$mi\f$ is the min_bound and \f$ma\f$ is the max_bound of
    /// the axis aligned bounding box, and \f$s\f$ and \f$c\f$ are the
    /// provided scaling factor and center respectively, then the new
    /// min_bound and max_bound are given by \f$mi = c + s (mi - c)\f$
    /// and \f$ma = c + s (ma - c)\f$.
    ///
    /// \param scale The scale parameter.
    /// \param center Center used for the scaling operation.
    virtual AxisAlignedBoundingBox& Scale(
            const double scale, const Eigen::Vector3d& center) override;

    /// \brief an AxisAlignedBoundingBox can not be rotated. This method
    /// will throw an error.
    virtual AxisAlignedBoundingBox& Rotate(
            const Eigen::Matrix3d& R, const Eigen::Vector3d& center) override;

    AxisAlignedBoundingBox& operator+=(const AxisAlignedBoundingBox& other);

    /// Get the extent/length of the bounding box in x, y, and z dimension.
    Eigen::Vector3d GetExtent() const { return (max_bound_ - min_bound_); }

    /// Returns the half extent of the bounding box.
    Eigen::Vector3d GetHalfExtent() const { return GetExtent() * 0.5; }

    /// Returns the maximum extent, i.e. the maximum of X, Y and Z axis'
    /// extents.
    double GetMaxExtent() const { return (max_bound_ - min_bound_).maxCoeff(); }

    /// Calculates the percentage position of the given x-coordinate within
    /// the x-axis range of this AxisAlignedBoundingBox.
    double GetXPercentage(double x) const {
        return (x - min_bound_(0)) / (max_bound_(0) - min_bound_(0));
    }

    /// Calculates the percentage position of the given y-coordinate within
    /// the y-axis range of this AxisAlignedBoundingBox.
    double GetYPercentage(double y) const {
        return (y - min_bound_(1)) / (max_bound_(1) - min_bound_(1));
    }

    /// Calculates the percentage position of the given z-coordinate within
    /// the z-axis range of this AxisAlignedBoundingBox.
    double GetZPercentage(double z) const {
        return (z - min_bound_(2)) / (max_bound_(2) - min_bound_(2));
    }

    /// Returns the volume of the bounding box.
    double Volume() const;
    /// Returns the eight points that define the bounding box.
    std::vector<Eigen::Vector3d> GetBoxPoints() const;

    /// Return indices to points that are within the bounding box.
    ///
    /// \param points A list of points.
    std::vector<size_t> GetPointIndicesWithinBoundingBox(
            const std::vector<Eigen::Vector3d>& points) const;

    /// Returns the 3D dimensions of the bounding box in string format.
    std::string GetPrintInfo() const;

    /// Creates the bounding box that encloses the set of points.
    ///
    /// \param points A list of points.
    static AxisAlignedBoundingBox CreateFromPoints(
            const std::vector<Eigen::Vector3d>& points);

public:
    /// The lower x, y, z bounds of the bounding box.
    Eigen::Vector3d min_bound_;
    /// The upper x, y, z bounds of the bounding box.
    Eigen::Vector3d max_bound_;
    /// The color of the bounding box in RGB.
    Eigen::Vector3d color_;
};

}  // namespace geometry
}  // namespace open3d
