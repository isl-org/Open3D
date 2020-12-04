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

#include "open3d/t/geometry/PointCloud.h"

#include <Eigen/Core>
#include <string>
#include <unordered_map>

#include "open3d/core/EigenConverter.h"
#include "open3d/core/ShapeUtil.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/TensorMap.h"

namespace open3d {
namespace t {
namespace geometry {

PointCloud::PointCloud(core::Dtype dtype, const core::Device &device)
    : Geometry(Geometry::GeometryType::PointCloud, 3),
      device_(device),
      point_attr_(TensorMap("points")) {
    SetPoints(core::Tensor::Zeros({0, 3}, dtype, device_));
}

PointCloud::PointCloud(const core::Tensor &points)
    : PointCloud(points.GetDtype(), points.GetDevice()) {
    points.AssertShapeCompatible({utility::nullopt, 3});
    SetPoints(points);
}

PointCloud::PointCloud(const std::unordered_map<std::string, core::Tensor>
                               &map_keys_to_tensors)
    : PointCloud(map_keys_to_tensors.at("points").GetDtype(),
                 map_keys_to_tensors.at("points").GetDevice()) {
    if (map_keys_to_tensors.count("points") == 0) {
        utility::LogError("\"points\" attribute must be specified.");
    }
    map_keys_to_tensors.at("points").AssertShapeCompatible(
            {utility::nullopt, 3});
    point_attr_ = TensorMap("points", map_keys_to_tensors.begin(),
                            map_keys_to_tensors.end());
}

core::Tensor PointCloud::GetMinBound() const { return GetPoints().Min({0}); }

core::Tensor PointCloud::GetMaxBound() const { return GetPoints().Max({0}); }

core::Tensor PointCloud::GetCenter() const { return GetPoints().Mean({0}); }

PointCloud &PointCloud::Transform(const core::Tensor &transformation) {
    transformation.AssertShape({4, 4});
    // Use reserve() keyword.
    core::Tensor transform;
    if (transformation.GetDevice() != device_) {
        utility::LogWarning("Attribute device {} != Pointcloud's device {}.",
                            transformation.GetDevice().ToString(),
                            device_.ToString());
        transform = transformation.Copy(device_);  // Copy to the device.
    } else
        transform = transformation;  // Use Shallow Copy, if on the same device.

    core::Tensor &points = GetPoints();

    //  Extract s, R, t from Transformation
    //  T (4x4) =   | R(3x3)  t(3x1) |
    //              | O(1x3)  s(1x1) |  for Rigid Transformation s = 1

    core::Tensor R = transform.Slice(0, 0, 3).Slice(1, 0, 3);
    core::Tensor t = transform.Slice(0, 0, 3).Slice(1, 3, 4);

    // To be considered for future opitmisation:
    // - Since rotation operation is common for both points and normals
    //   in future a parallel joint optimised kernel can be defined.
    //   [Cache Optimisation]
    // - After tensor: vertical stack operation is implemented,
    //   performance of P = T*P vs P = R(P) + t is to be compared
    //   for sequencial and parallel calculation.

    //  points (3xN) = s.R(points) + t
    points = (R.Matmul(points.T())).Add_(t).T();

    if (HasPointNormals()) {
        // for normal: n.T() = R*n.T()
        core::Tensor &normals = GetPointNormals();
        normals = (R.Matmul(normals.T())).T();
    }
    return *this;
}

PointCloud &PointCloud::Translate(const core::Tensor &translation,
                                  bool relative) {
    translation.AssertShape({3});
    // Use reserve() keyword.
    core::Tensor transform;
    if (translation.GetDevice() != device_) {
        utility::LogWarning("Attribute device {} != Pointcloud's device {}.",
                            translation.GetDevice().ToString(),
                            device_.ToString());
        transform = translation.Copy(device_);  // Copy to 'this' device.
    } else
        transform = translation;  // Use Shallow Copy, if on the same device.

    if (!relative) {
        transform -= GetCenter();
    }
    GetPoints() += transform;
    return *this;
}

PointCloud &PointCloud::Scale(double scale, const core::Tensor &center) {
    center.AssertShape({3});
    core::Tensor points = GetPoints();
    points.Sub_(center).Mul_(scale).Add_(center);
    return *this;
}

PointCloud &PointCloud::Rotate(const core::Tensor &R,
                               const core::Tensor &center) {
    R.AssertShape({3, 3});
    center.AssertShape({3});
    // Use reserve() keyword.
    core::Tensor Rot;
    if (R.GetDevice() != device_) {
        utility::LogWarning("Attribute device {} != Pointcloud's device {}.",
                            R.GetDevice().ToString(), device_.ToString());
        Rot = R.Copy(device_);  // Copy to 'this' device.
    } else
        Rot = R;  // Use Shallow Copy, if on the same device.

    core::Tensor &points = GetPoints();

    // Doubt: if center is 0, will it still perform substration computationally?
    // if so, crete case for zero and non-zero to save unnecessary computation,
    // or create a new function 'RotateAboutOrigin'
    points = ((Rot.Matmul((points.Sub_(center)).T())).T()).Add_(center);

    if (HasPointNormals()) {
        // for normal: n.T() = R*n.T()
        core::Tensor &normals = GetPointNormals();
        normals = (Rot.Matmul(normals.T())).T();
    }

    // utility::LogError("Unimplemented");
    return *this;
}

geometry::PointCloud PointCloud::FromLegacyPointCloud(
        const open3d::geometry::PointCloud &pcd_legacy,
        core::Dtype dtype,
        const core::Device &device) {
    geometry::PointCloud pcd(dtype, device);
    if (pcd_legacy.HasPoints()) {
        pcd.SetPoints(core::eigen_converter::EigenVector3dVectorToTensor(
                pcd_legacy.points_, dtype, device));
    } else {
        utility::LogWarning(
                "Creating from an empty legacy pointcloud, an empty pointcloud "
                "with default dtype and device will be created.");
    }
    if (pcd_legacy.HasColors()) {
        pcd.SetPointColors(core::eigen_converter::EigenVector3dVectorToTensor(
                pcd_legacy.colors_, dtype, device));
    }
    if (pcd_legacy.HasNormals()) {
        pcd.SetPointNormals(core::eigen_converter::EigenVector3dVectorToTensor(
                pcd_legacy.normals_, dtype, device));
    }
    return pcd;
}

open3d::geometry::PointCloud PointCloud::ToLegacyPointCloud() const {
    open3d::geometry::PointCloud pcd_legacy;
    if (HasPoints()) {
        const core::Tensor &points = GetPoints();
        pcd_legacy.points_.reserve(points.GetLength());
        for (int64_t i = 0; i < points.GetLength(); i++) {
            pcd_legacy.points_.push_back(
                    core::eigen_converter::TensorToEigenVector3d(points[i]));
        }
    }
    if (HasPointColors()) {
        const core::Tensor &colors = GetPointColors();
        pcd_legacy.colors_.reserve(colors.GetLength());
        for (int64_t i = 0; i < colors.GetLength(); i++) {
            pcd_legacy.colors_.push_back(
                    core::eigen_converter::TensorToEigenVector3d(colors[i]));
        }
    }
    if (HasPointNormals()) {
        const core::Tensor &normals = GetPointNormals();
        pcd_legacy.normals_.reserve(normals.GetLength());
        for (int64_t i = 0; i < normals.GetLength(); i++) {
            pcd_legacy.normals_.push_back(
                    core::eigen_converter::TensorToEigenVector3d(normals[i]));
        }
    }
    return pcd_legacy;
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
