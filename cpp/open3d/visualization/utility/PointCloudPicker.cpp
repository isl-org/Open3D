// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/utility/PointCloudPicker.h"

#include "open3d/geometry/BoundingVolume.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace visualization {

PointCloudPicker& PointCloudPicker::Clear() {
    picked_indices_.clear();
    return *this;
}

bool PointCloudPicker::IsEmpty() const {
    return (!pointcloud_ptr_ || picked_indices_.empty());
}

Eigen::Vector3d PointCloudPicker::GetMinBound() const {
    if (pointcloud_ptr_) {
        return ((const geometry::PointCloud&)(*pointcloud_ptr_)).GetMinBound();
    } else {
        return Eigen::Vector3d(0.0, 0.0, 0.0);
    }
}

Eigen::Vector3d PointCloudPicker::GetMaxBound() const {
    if (pointcloud_ptr_) {
        return ((const geometry::PointCloud&)(*pointcloud_ptr_)).GetMaxBound();
    } else {
        return Eigen::Vector3d(0.0, 0.0, 0.0);
    }
}

Eigen::Vector3d PointCloudPicker::GetCenter() const {
    if (pointcloud_ptr_) {
        return ((const geometry::PointCloud&)(*pointcloud_ptr_)).GetCenter();
    } else {
        return Eigen::Vector3d(0.0, 0.0, 0.0);
    }
}

geometry::AxisAlignedBoundingBox PointCloudPicker::GetAxisAlignedBoundingBox()
        const {
    if (pointcloud_ptr_) {
        return geometry::AxisAlignedBoundingBox::CreateFromPoints(
                ((const geometry::PointCloud&)(*pointcloud_ptr_)).points_);
    } else {
        return geometry::AxisAlignedBoundingBox();
    }
}

geometry::OrientedBoundingBox PointCloudPicker::GetOrientedBoundingBox(
        bool robust) const {
    if (pointcloud_ptr_) {
        return geometry::OrientedBoundingBox::CreateFromPoints(
                ((const geometry::PointCloud&)(*pointcloud_ptr_)).points_,
                robust);
    } else {
        return geometry::OrientedBoundingBox();
    }
}

geometry::OrientedBoundingBox PointCloudPicker::GetMinimalOrientedBoundingBox(
        bool robust) const {
    if (pointcloud_ptr_) {
        return geometry::OrientedBoundingBox::CreateFromPointsMinimal(
                ((const geometry::PointCloud&)(*pointcloud_ptr_)).points_,
                robust);
    } else {
        return geometry::OrientedBoundingBox();
    }
}

PointCloudPicker& PointCloudPicker::Transform(
        const Eigen::Matrix4d& /*transformation*/) {
    // Do nothing
    return *this;
}

PointCloudPicker& PointCloudPicker::Translate(
        const Eigen::Vector3d& translation, bool relative) {
    // Do nothing
    return *this;
}

PointCloudPicker& PointCloudPicker::Scale(const double scale,
                                          const Eigen::Vector3d& center) {
    // Do nothing
    return *this;
}

PointCloudPicker& PointCloudPicker::Rotate(const Eigen::Matrix3d& R,
                                           const Eigen::Vector3d& center) {
    // Do nothing
    return *this;
}

bool PointCloudPicker::SetPointCloud(
        std::shared_ptr<const geometry::Geometry> ptr) {
    if (!ptr || ptr->GetGeometryType() !=
                        geometry::Geometry::GeometryType::PointCloud) {
        return false;
    }
    pointcloud_ptr_ = ptr;
    return true;
}

}  // namespace visualization
}  // namespace open3d
