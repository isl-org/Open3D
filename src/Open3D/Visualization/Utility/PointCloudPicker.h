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

#include "Open3D/Geometry/Geometry3D.h"

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
    geometry::AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const final;
    geometry::OrientedBoundingBox GetOrientedBoundingBox() const final;
    PointCloudPicker& Transform(const Eigen::Matrix4d& transformation) override;
    PointCloudPicker& Translate(const Eigen::Vector3d& translation,
                                bool relative = true) override;
    PointCloudPicker& Scale(const double scale, bool center = true) override;
    PointCloudPicker& Rotate(const Eigen::Matrix3d& R,
                             bool center = true) override;
    bool SetPointCloud(std::shared_ptr<const geometry::Geometry> ptr);

public:
    std::shared_ptr<const geometry::Geometry> pointcloud_ptr_;
    std::vector<size_t> picked_indices_;
};

}  // namespace visualization
}  // namespace open3d
