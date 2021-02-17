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
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "open3d/core/Tensor.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/t/geometry/Geometry.h"
#include "open3d/t/geometry/Image.h"
#include "open3d/t/geometry/TensorMap.h"
#include "open3d/utility/Console.h"

namespace open3d {
namespace t {
namespace geometry {

/// \class PointCloud
/// \brief A pointcloud contains a set of 3D points.
///
/// The PointCloud class stores the attribute data in key-value pairs for
/// flexibility, where the key is a string representing the attribute name and
/// value is a Tensor containing the attribute data. In most cases, the
/// length of an attribute should be equal to the length of the "points", the
/// pointcloud class provides helper functions to check and facilitates this
/// consistency.
///
/// Although the attributes are all stored in a key-value pair dictionary, the
/// attributes have different levels:
///
/// - Level 0: Default attribute {"points"}.
///     - Created by default, required for all pointclouds.
///     - The tensor must be of shape N x {3,}.
///     - Convenience functions:
///         - PointCloud::GetPoints()
///         - PointCloud::SetPoints(points_tensor)
///         - PointCloud::HasPoints()
///     - The device of "points" determines the device of the pointcloud.
/// - Level 1: Commonly-used attributes {"normals", "colors"}.
///     - Not created by default.
///     - The tensor must be of shape N x {3,}.
///     - Convenience functions:
///         - PointCloud::GetPointNormals()
///         - PointCloud::SetPointNormals(normals_tensor)
///         - PointCloud::HasPointNormals()
///         - PointCloud::GetPointColors()
///         - PointCloud::SetPointColors(colors_tensor)
///         - PointCloud::HasPointColors()
///     - Device must be the same as the device of "points". Dtype can be
///       different.
/// - Level 2: Custom attributes, e.g. {"labels", "alphas", "intensities"}.
///     - Not created by default. Created by users.
///     - No convenience functions.
///     - Use generalized helper functions. Examples:
///         - PointCloud::GetPointAttr("labels")
///         - PointCloud::SetPointAttr("labels", labels_tensor)
///         - PointCloud::HasPointAttr("labels")
///     - Device must be the same as the device of "points". Dtype can be
///       different.
///
/// Note that the level 0 and level 1 convenience functions can also be achieved
/// via the generalized helper functions:
///     - PointCloud::GetPoints() is the same as
///       PointCloud::GetPointAttr("points")
///     - PointCloud::HasPointNormals() is the same as
///       PointCloud::HasPointAttr("normals")
class PointCloud : public Geometry {
public:
    /// Construct an empty pointcloud.
    PointCloud(const core::Device &device = core::Device("CPU:0"));

    /// Construct a pointcloud from points.
    ///
    /// \param points A tensor with element shape (3,).
    /// - The resulting pointcloud will have the same dtype and device as the
    /// tensor.
    /// - If the tensor is created in-place from a pre-allocated buffer, the
    /// tensor has a fixed size and thus the resulting pointcloud will have
    /// a fixed size and calling to functions like `SynchronizedPushBack` will
    /// raise an exception.
    PointCloud(const core::Tensor &points);

    /// Construct from points and other attributes of the points.
    ///
    /// \param map_keys_to_tensors A map of string to Tensor containing
    /// points and their attributes. point_dict must contain at least the
    /// "points" key.
    PointCloud(const std::unordered_map<std::string, core::Tensor>
                       &map_keys_to_tensors);

    virtual ~PointCloud() override {}

    /// Getter for point_attr_ TensorMap. Used in Pybind.
    const TensorMap &GetPointAttr() const { return point_attr_; }

    /// Get attributes. Throws exception if the attribute does not exist.
    ///
    /// \param key Attribute name.
    core::Tensor &GetPointAttr(const std::string &key) {
        return point_attr_.at(key);
    }

    /// Get the value of the "points" attribute. Convenience function.
    core::Tensor &GetPoints() { return GetPointAttr("points"); }

    /// Get the value of the "colors" attribute. Convenience function.
    core::Tensor &GetPointColors() { return GetPointAttr("colors"); }

    /// Get the value of the "normals" attribute. Convenience function.
    core::Tensor &GetPointNormals() { return GetPointAttr("normals"); }

    /// Get attributes. Throws exception if the attribute does not exist.
    ///
    /// \param key Attribute name.
    const core::Tensor &GetPointAttr(const std::string &key) const {
        return point_attr_.at(key);
    }

    /// Get the value of the "points" attribute. Convenience function.
    const core::Tensor &GetPoints() const { return GetPointAttr("points"); }

    /// Get the value of the "colors" attribute. Convenience function.
    const core::Tensor &GetPointColors() const {
        return GetPointAttr("colors");
    }

    /// Get the value of the "normals" attribute. Convenience function.
    const core::Tensor &GetPointNormals() const {
        return GetPointAttr("normals");
    }

    /// Set attributes. If the attribute key already exists, its value
    /// will be overwritten, otherwise, the new key will be created.
    ///
    /// \param key Attribute name.
    /// \param value A tensor.
    void SetPointAttr(const std::string &key, const core::Tensor &value) {
        if (value.GetDevice() != device_) {
            utility::LogError("Attribute device {} != Pointcloud's device {}.",
                              value.GetDevice().ToString(), device_.ToString());
        }
        point_attr_[key] = value;
    }

    /// Set the value of the "points" attribute. Convenience function.
    void SetPoints(const core::Tensor &value) {
        value.AssertShapeCompatible({utility::nullopt, 3});
        SetPointAttr("points", value);
    }

    /// Set the value of the "colors" attribute. Convenience function.
    void SetPointColors(const core::Tensor &value) {
        value.AssertShapeCompatible({utility::nullopt, 3});
        SetPointAttr("colors", value);
    }

    /// Set the value of the "normals" attribute. Convenience function.
    void SetPointNormals(const core::Tensor &value) {
        value.AssertShapeCompatible({utility::nullopt, 3});
        SetPointAttr("normals", value);
    }

    /// Returns true if all of the followings are true:
    /// 1) attribute key exist
    /// 2) attribute's length as points' length
    /// 3) attribute's length > 0
    bool HasPointAttr(const std::string &key) const {
        return point_attr_.Contains(key) && GetPointAttr(key).GetLength() > 0 &&
               GetPointAttr(key).GetLength() == GetPoints().GetLength();
    }

    /// Check if the "points" attribute's value has length > 0.
    /// This is a convenience function.
    bool HasPoints() const { return HasPointAttr("points"); }

    /// Returns true if all of the followings are true:
    /// 1) attribute "colors" exist
    /// 2) attribute "colors"'s length as points' length
    /// 3) attribute "colors"'s length > 0
    /// This is a convenience function.
    bool HasPointColors() const { return HasPointAttr("colors"); }

    /// Returns true if all of the followings are true:
    /// 1) attribute "normals" exist
    /// 2) attribute "normals"'s length as points' length
    /// 3) attribute "normals"'s length > 0
    /// This is a convenience function.
    bool HasPointNormals() const { return HasPointAttr("normals"); }

public:
    /// Transfer the point cloud to a specified device.
    /// \param device The targeted device to convert to.
    /// \param copy If true, a new point cloud is always created; if false, the
    /// copy is avoided when the original point cloud is already on the targeted
    /// device.
    PointCloud To(const core::Device &device, bool copy = false) const;

    /// Returns copy of the point cloud on the same device.
    PointCloud Clone() const;

    /// Transfer the point cloud to CPU.
    ///
    /// If the point cloud is already on CPU, no copy will be performed.
    PointCloud CPU() const { return To(core::Device("CPU:0")); };

    /// Transfer the point cloud to a CUDA device.
    ///
    /// If the point cloud is already on the specified CUDA device, no copy will
    /// be performed.
    PointCloud CUDA(int device_id = 0) const {
        return To(core::Device(core::Device::DeviceType::CUDA, device_id));
    };

    /// Clear all data in the pointcloud.
    PointCloud &Clear() override {
        point_attr_.clear();
        return *this;
    }

    /// Returns !HasPoints().
    bool IsEmpty() const override { return !HasPoints(); }

    /// Returns the min bound for point coordinates.
    core::Tensor GetMinBound() const;

    /// Returns the max bound for point coordinates.
    core::Tensor GetMaxBound() const;

    /// Returns the center for point coordinates.
    core::Tensor GetCenter() const;

    /// \brief Transforms the points and normals (if exist)
    /// of the PointCloud.
    /// Extracts R, t from Transformation
    ///  T (4x4) =   [[ R(3x3)  t(3x1) ],
    ///               [ O(1x3)  s(1x1) ]]
    ///  (s = 1 for Transformation wihtout scaling)
    /// PS. It Assumes s = 1 and O = [0,0,0]
    /// and applies the transformation as P = R(P) + t
    /// \param transformation Transformation [Tensor of dim {4,4}].
    /// Should be on the same device as the PointCloud
    /// \return Transformed pointcloud
    PointCloud &Transform(const core::Tensor &transformation);

    /// \brief Translates the points of the PointCloud.
    /// \param translation translation tensor of dimention {3}
    /// Should be on the same device as the PointCloud
    /// \param relative if true (default): translates relative to Center
    /// \return Translated pointcloud
    PointCloud &Translate(const core::Tensor &translation,
                          bool relative = true);

    /// \brief Scales the points of the PointCloud.
    /// \param scale Scale [double] of dimention
    /// \param center Center [Tensor of dim {3}] about which the PointCloud is
    /// to be scaled. Should be on the same device as the PointCloud
    /// \return Scaled pointcloud
    PointCloud &Scale(double scale, const core::Tensor &center);

    /// \brief Rotates the points and normals (if exists).
    /// \param R Rotation [Tensor of dim {3,3}].
    /// Should be on the same device as the PointCloud
    /// \param center Center [Tensor of dim {3}] about which the PointCloud is
    /// to be scaled. Should be on the same device as the PointCloud
    /// \return Rotated pointcloud
    PointCloud &Rotate(const core::Tensor &R, const core::Tensor &center);

    /// \brief Returns the device attribute of this PointCloud.
    core::Device GetDevice() const { return device_; }

    /// \brief Factory function to create a pointcloud from a depth image and a
    /// camera model.
    ///
    /// Given depth value d at (u, v) image coordinate, the corresponding 3d
    /// point is: z = d / depth_scale\n x = (u - cx) * z / fx\n y = (v - cy) * z
    /// / fy\n
    ///
    /// \param depth The input depth image should be a uint16_t image.
    /// \param intrinsic Intrinsic parameters of the camera.
    /// \param extrinsic Extrinsic parameters of the camera.
    /// \param depth_scale The depth is scaled by 1 / \p depth_scale.
    /// \param depth_trunc Truncated at \p depth_trunc distance.
    /// \param stride Sampling factor to support coarse point cloud extraction.
    ///
    /// \return An empty pointcloud if the conversion fails.
    /// If \param project_valid_depth_only is true, return point cloud, which
    /// doesn't
    /// have nan point. If the value is false, return point cloud, which has
    /// a point for each pixel, whereas invalid depth results in NaN points.
    static PointCloud CreateFromDepthImage(
            const Image &depth,
            const core::Tensor &intrinsics,
            const core::Tensor &extrinsics = core::Tensor::Eye(
                    4, core::Dtype::Float32, core::Device("CPU:0")),
            float depth_scale = 1000.0f,
            float depth_max = 3.0f,
            int stride = 1);

    /// Create a PointCloud from a legacy Open3D PointCloud.
    static PointCloud FromLegacyPointCloud(
            const open3d::geometry::PointCloud &pcd_legacy,
            core::Dtype dtype = core::Dtype::Float32,
            const core::Device &device = core::Device("CPU:0"));

    /// Convert to a legacy Open3D PointCloud.
    open3d::geometry::PointCloud ToLegacyPointCloud() const;

protected:
    core::Device device_ = core::Device("CPU:0");
    TensorMap point_attr_;
};

}  // namespace geometry
}  // namespace t
}  // namespace open3d
