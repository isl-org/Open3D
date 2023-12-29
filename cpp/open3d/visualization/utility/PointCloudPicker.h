// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
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
}
namespace visualization {

/// A utility class to store picked points of a pointcloud
class PointCloudPicker : public geometry::Geometry3D {
public:
    PointCloudPicker()
        : geometry::Geometry3D(geometry::Geometry::GeometryType::Unspecified) {}
    ~PointCloudPicker() override {}

public:
    PointCloudPicker& Clear() override;
    bool IsEmpty() const override;
    Eigen::Vector3d GetMinBound() const final;
    Eigen::Vector3d GetMaxBound() const final;
    Eigen::Vector3d GetCenter() const final;

    /// If point cloud does not exist, creates an axis-aligned bounding box
    /// form its default constructor. If point cloud exists, creates the
    /// axis-aligned bounding box around it. Further details in
    /// AxisAlignedBoundingBox::CreateFromPoints()
    geometry::AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const final;

    /// If point cloud does not exist, creates an oriented bounding box
    /// form its default constructor. If point cloud exists, creates the
    /// oriented bounding box around it. Further details in
    /// OrientedBoundingBox::CreateFromPoints()
    /// \param robust If set to true uses a more robust method which works
    ///               in degenerate cases but introduces noise to the points
    ///               coordinates.
    geometry::OrientedBoundingBox GetOrientedBoundingBox(
            bool robust = false) const final;

    /// If point cloud does not exist, creates an oriented bounding box
    /// form its default constructor. If point cloud exists, creates the minimal
    /// oriented bounding box around it.
    /// Further details in OrientedBoundingBox::CreateFromPointsMinimal()
    /// \param robust If set to true uses a more robust method which works
    ///               in degenerate cases but introduces noise to the points
    ///               coordinates.
    geometry::OrientedBoundingBox GetMinimalOrientedBoundingBox(
            bool robust = false) const final;

    PointCloudPicker& Transform(const Eigen::Matrix4d& transformation) override;
    PointCloudPicker& Translate(const Eigen::Vector3d& translation,
                                bool relative = true) override;
    PointCloudPicker& Scale(const double scale,
                            const Eigen::Vector3d& center) override;
    PointCloudPicker& Rotate(const Eigen::Matrix3d& R,
                             const Eigen::Vector3d& center) override;
    bool SetPointCloud(std::shared_ptr<const geometry::Geometry> ptr);

public:
    std::shared_ptr<const geometry::Geometry> pointcloud_ptr_;
    std::vector<size_t> picked_indices_;
};

}  // namespace visualization
}  // namespace open3d
