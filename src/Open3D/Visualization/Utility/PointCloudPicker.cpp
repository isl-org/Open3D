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

#include "Open3D/Visualization/Utility/PointCloudPicker.h"

#include "Open3D/Geometry/BoundingVolume.h"
#include "Open3D/Geometry/PointCloud.h"
#include "Open3D/Utility/Console.h"

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

geometry::OrientedBoundingBox PointCloudPicker::GetOrientedBoundingBox() const {
    if (pointcloud_ptr_) {
        return geometry::OrientedBoundingBox::CreateFromPoints(
                ((const geometry::PointCloud&)(*pointcloud_ptr_)).points_);
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

PointCloudPicker& PointCloudPicker::Scale(const double scale, bool center) {
    // Do nothing
    return *this;
}

PointCloudPicker& PointCloudPicker::Rotate(const Eigen::Matrix3d& R,
                                           bool center) {
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
