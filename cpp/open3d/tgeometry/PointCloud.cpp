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

#include "open3d/tgeometry/PointCloud.h"

#include <Eigen/Core>
#include <string>
#include <unordered_map>

#include "open3d/core/EigenConverter.h"
#include "open3d/core/ShapeUtil.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/TensorList.h"

namespace open3d {
namespace tgeometry {

PointCloud::PointCloud(core::Dtype dtype, const core::Device &device)
    : Geometry(Geometry::GeometryType::PointCloud, 3),
      device_(device),
      point_attr_(TensorListMap("points")) {
    SetPoints(core::TensorList({3}, dtype, device_));
}

PointCloud::PointCloud(const core::TensorList &points)
    : PointCloud(points.GetDtype(), points.GetDevice()) {
    points.AssertElementShape({3});
    SetPoints(points);
}

PointCloud::PointCloud(const std::unordered_map<std::string, core::TensorList>
                               &map_keys_to_tensorlists)
    : PointCloud(map_keys_to_tensorlists.at("points").GetDtype(),
                 map_keys_to_tensorlists.at("points").GetDevice()) {
    map_keys_to_tensorlists.at("points").AssertElementShape({3});
    point_attr_.Assign(map_keys_to_tensorlists);
}

void PointCloud::SynchronizedPushBack(
        const std::unordered_map<std::string, core::Tensor>
                &map_keys_to_tensors) {
    point_attr_.SynchronizedPushBack(map_keys_to_tensors);
}

PointCloud &PointCloud::Clear() {
    point_attr_.clear();
    return *this;
}

bool PointCloud::IsEmpty() const { return !HasPoints(); }

core::Tensor PointCloud::GetMinBound() const {
    return GetPoints().AsTensor().Min({0});
}

core::Tensor PointCloud::GetMaxBound() const {
    return GetPoints().AsTensor().Max({0});
}

core::Tensor PointCloud::GetCenter() const {
    return GetPoints().AsTensor().Mean({0});
}

PointCloud &PointCloud::Transform(const core::Tensor &transformation) {
    utility::LogError("Unimplemented");
    return *this;
}

PointCloud &PointCloud::Translate(const core::Tensor &translation,
                                  bool relative) {
    translation.AssertShape({3});
    core::Tensor transform = translation.Copy();
    if (!relative) {
        transform -= GetCenter();
    }
    GetPoints().AsTensor() += transform;
    return *this;
}

PointCloud &PointCloud::Scale(double scale, const core::Tensor &center) {
    center.AssertShape({3});
    core::Tensor points = GetPoints().AsTensor();
    points.Sub_(center).Mul_(scale).Add_(center);
    return *this;
}

PointCloud &PointCloud::Rotate(const core::Tensor &R,
                               const core::Tensor &center) {
    utility::LogError("Unimplemented");
    return *this;
}

tgeometry::PointCloud PointCloud::FromLegacyPointCloud(
        const geometry::PointCloud &pcd_legacy,
        core::Dtype dtype,
        const core::Device &device) {
    tgeometry::PointCloud pcd(dtype, device);
    if (pcd_legacy.HasPoints()) {
        pcd.GetPoints() = core::TensorList({3}, dtype, device);
        // TODO: optimization always create host tensorlist first, and then copy
        // to device.
        for (const Eigen::Vector3d &point : pcd_legacy.points_) {
            pcd.GetPoints().PushBack(
                    core::eigen_converter::EigenVector3dToTensor(point, dtype,
                                                                 device));
        }
    } else {
        utility::LogWarning(
                "Creating from an empty legacy pointcloud, an empty pointcloud "
                "with default dtype and device will be created.");
    }
    if (pcd_legacy.HasColors()) {
        pcd.SetPointColors(core::TensorList({3}, dtype, device));
        for (const Eigen::Vector3d &color : pcd_legacy.colors_) {
            pcd.GetPointColors().PushBack(
                    core::eigen_converter::EigenVector3dToTensor(color, dtype,
                                                                 device));
        }
    }
    if (pcd_legacy.HasNormals()) {
        pcd.SetPointNormals(core::TensorList({3}, dtype, device));
        for (const Eigen::Vector3d &normal : pcd_legacy.normals_) {
            pcd.GetPointNormals().PushBack(
                    core::eigen_converter::EigenVector3dToTensor(normal, dtype,
                                                                 device));
        }
    }
    return pcd;
}

geometry::PointCloud PointCloud::ToLegacyPointCloud() const {
    geometry::PointCloud pcd_legacy;
    if (HasPoints()) {
        const core::TensorList &points = GetPoints();
        for (int64_t i = 0; i < points.GetSize(); i++) {
            pcd_legacy.points_.push_back(
                    core::eigen_converter::TensorToEigenVector3d(points[i]));
        }
    }
    if (HasPointColors()) {
        const core::TensorList &colors = GetPointColors();
        for (int64_t i = 0; i < colors.GetSize(); i++) {
            pcd_legacy.colors_.push_back(
                    core::eigen_converter::TensorToEigenVector3d(colors[i]));
        }
    }
    if (HasPointNormals()) {
        const core::TensorList &normals = GetPointNormals();
        for (int64_t i = 0; i < normals.GetSize(); i++) {
            pcd_legacy.normals_.push_back(
                    core::eigen_converter::TensorToEigenVector3d(normals[i]));
        }
    }
    return pcd_legacy;
}

}  // namespace tgeometry
}  // namespace open3d
