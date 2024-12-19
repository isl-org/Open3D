// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <memory>
#include <vector>

#include "open3d/geometry/Geometry3D.h"

namespace open3d {
namespace geometry {

class PointCloud;
class OrientedBoundingBox;
class AxisAlignedBoundingBox;
class TriangleMesh;
class TetraMesh;

/// \class LineSet
///
/// \brief LineSet define a sets of lines in 3D. A typical application is to
/// display the point cloud correspondence pairs.
class LineSet : public Geometry3D {
public:
    /// \brief Default Constructor.
    LineSet() : Geometry3D(Geometry::GeometryType::LineSet) {}
    /// \brief Parameterized Constructor.
    ///
    ///  Create a LineSet from given points and line indices
    ///
    /// \param points Point coordinates.
    /// \param lines Lines denoted by the index of points forming the line.
    LineSet(const std::vector<Eigen::Vector3d> &points,
            const std::vector<Eigen::Vector2i> &lines)
        : Geometry3D(Geometry::GeometryType::LineSet),
          points_(points),
          lines_(lines) {}
    ~LineSet() override {}

public:
    LineSet &Clear() override;
    bool IsEmpty() const override;
    Eigen::Vector3d GetMinBound() const override;
    Eigen::Vector3d GetMaxBound() const override;
    Eigen::Vector3d GetCenter() const override;

    /// Creates the axis-aligned bounding box around the points of the object.
    /// Further details in AxisAlignedBoundingBox::CreateFromPoints()
    AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const override;

    /// Creates an oriented bounding box around the points of the object.
    /// Further details in OrientedBoundingBox::CreateFromPoints()
    /// \param robust If set to true uses a more robust method which works
    ///               in degenerate cases but introduces noise to the points
    ///               coordinates.
    OrientedBoundingBox GetOrientedBoundingBox(
            bool robust = false) const override;

    /// Creates the minimal oriented bounding box around the points of the
    /// object. Further details in
    /// OrientedBoundingBox::CreateFromPointsMinimal()
    /// \param robust If set to true uses a more robust method which works
    ///               in degenerate cases but introduces noise to the points
    ///               coordinates.
    OrientedBoundingBox GetMinimalOrientedBoundingBox(
            bool robust = false) const override;

    LineSet &Transform(const Eigen::Matrix4d &transformation) override;
    LineSet &Translate(const Eigen::Vector3d &translation,
                       bool relative = true) override;
    LineSet &Scale(const double scale, const Eigen::Vector3d &center) override;
    LineSet &Rotate(const Eigen::Matrix3d &R,
                    const Eigen::Vector3d &center) override;

    LineSet &operator+=(const LineSet &lineset);
    LineSet operator+(const LineSet &lineset) const;

    /// Returns `true` if the object contains points.
    bool HasPoints() const { return points_.size() > 0; }

    /// Returns `true` if the object contains lines.
    bool HasLines() const { return HasPoints() && lines_.size() > 0; }

    /// Returns `true` if the objects lines contains colors.
    bool HasColors() const {
        return HasLines() && colors_.size() == lines_.size();
    }

    /// \brief Returns the coordinates of the line at the given index.
    ///
    /// \param line_index Index of the line.
    std::pair<Eigen::Vector3d, Eigen::Vector3d> GetLineCoordinate(
            size_t line_index) const {
        return std::make_pair(points_[lines_[line_index][0]],
                              points_[lines_[line_index][1]]);
    }

    /// \brief Assigns each line in the LineSet the same color.
    ///
    /// \param color Specifies the color to be applied.
    LineSet &PaintUniformColor(const Eigen::Vector3d &color) {
        ResizeAndPaintUniformColor(colors_, lines_.size(), color);
        return *this;
    }

    /// \brief Factory function to create a LineSet from two PointClouds
    /// (\p cloud0, \p cloud1) and a correspondence set.
    ///
    /// \param cloud0 First point cloud.
    /// \param cloud1 Second point cloud.
    /// \param correspondences Set of correspondences.
    static std::shared_ptr<LineSet> CreateFromPointCloudCorrespondences(
            const PointCloud &cloud0,
            const PointCloud &cloud1,
            const std::vector<std::pair<int, int>> &correspondences);

    /// \brief Factory function to create a LineSet from an OrientedBoundingBox.
    ///
    /// \param box The input bounding box.
    static std::shared_ptr<LineSet> CreateFromOrientedBoundingBox(
            const OrientedBoundingBox &box);

    /// \brief Factory function to create a LineSet from an
    /// AxisAlignedBoundingBox.
    ///
    /// \param box The input bounding box.
    static std::shared_ptr<LineSet> CreateFromAxisAlignedBoundingBox(
            const AxisAlignedBoundingBox &box);

    /// Factory function to create a LineSet from edges of a triangle mesh.
    ///
    /// \param mesh The input triangle mesh.
    static std::shared_ptr<LineSet> CreateFromTriangleMesh(
            const TriangleMesh &mesh);

    /// Factory function to create a LineSet from edges of a tetra mesh.
    ///
    /// \param mesh The input tetra mesh.
    static std::shared_ptr<LineSet> CreateFromTetraMesh(const TetraMesh &mesh);

    /// Factory function to create a LineSet from intrinsic and extrinsic
    /// matrices.
    ///
    /// \param view_width_px The width of the view, in pixels
    /// \param view_height_px The height of the view, in pixels
    /// \param intrinsic The intrinsic matrix
    /// \param extrinsic The extrinsic matrix
    static std::shared_ptr<LineSet> CreateCameraVisualization(
            int view_width_px,
            int view_height_px,
            const Eigen::Matrix3d &intrinsic,
            const Eigen::Matrix4d &extrinsic,
            double scale = 1.0);

public:
    /// Points coordinates.
    std::vector<Eigen::Vector3d> points_;
    /// Lines denoted by the index of points forming the line.
    std::vector<Eigen::Vector2i> lines_;
    /// RGB colors of lines.
    std::vector<Eigen::Vector3d> colors_;
};

}  // namespace geometry
}  // namespace open3d
