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
#include "open3d/core/TensorList.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/tgeometry/Geometry.h"
#include "open3d/tgeometry/TensorListMap.h"

namespace open3d {
namespace tgeometry {

/// \class PointCloud
/// \brief A pointcloud contains a set of 3D points.
///
/// The PointCloud class stores the attribute data in key-value pairs for
/// flexibility, where the key is a string representing the attribute name and
/// value is a TensorList containing the attribute data. In most cases, the
/// length of an attribute should be equal to the length of the "points", the
/// pointcloud class provides helper functions to check and facilitates this
/// consistency.
///
/// Although the attributes are all stored in a key-value pair dictionary, the
/// attributes have different levels:
///
/// - Level 0: Default attribute {"points"}.
///     - Created by default, required for all pointclouds.
///     - The tensorlist must be of shape N x {3,}.
///     - Convenience functions:
///         - PointCloud::GetPoints()
///         - PointCloud::SetPoints(points_tensorlist)
///         - PointCloud::HasPoints()
///     - The device of "points" determines the device of the pointcloud.
/// - Level 1: Commonly-used attributes {"normals", "colors"}.
///     - Not created by default.
///     - The tensorlist must be of shape N x {3,}.
///     - Convenience functions:
///         - PointCloud::GetPointNormals()
///         - PointCloud::SetPointNormals(normals_tensorlist)
///         - PointCloud::HasPointNormals()
///         - PointCloud::GetPointColors()
///         - PointCloud::SetPointColors(colors_tensorlist)
///         - PointCloud::HasPointColors()
///     - Device must be the same as the device of "points". Dtype can be
///       different.
/// - Level 2: Custom attributes, e.g. {"labels", "alphas", "intensities"}.
///     - Not created by default. Created by users.
///     - No convenience functions.
///     - Use generalized helper functions. Examples:
///         - PointCloud::GetPointAttr("labels")
///         - PointCloud::SetPointAttr("labels", labels_tensorlist)
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
    PointCloud(core::Dtype dtype = core::Dtype::Float32,
               const core::Device &device = core::Device("CPU:0"));

    /// Construct a pointcloud from points.
    ///
    /// \param points A tensorlist with element shape (3,).
    /// - The resulting pointcloud will have the same dtype and device as the
    /// tensorlist.
    /// - If the tensorlist is created in-place from a pre-allocated buffer, the
    /// tensorlist has a fixed size and thus the resulting pointcloud will have
    /// a fixed size and calling to functions like `SynchronizedPushBack` will
    /// raise an exception.
    PointCloud(const core::TensorList &points);

    /// Construct from points and other attributes of the points.
    ///
    /// \param map_keys_to_tensorlists A map of string to TensorList containing
    /// points and their attributes. point_dict must contain at least the
    /// "points" key.
    PointCloud(const std::unordered_map<std::string, core::TensorList>
                       &map_keys_to_tensorlists);

    virtual ~PointCloud() override {}

    /// Get attributes. Throws exception if the attribute does not exist.
    ///
    /// \param key Attribute name.
    core::TensorList &GetPointAttr(const std::string &key) {
        return point_attr_.at(key);
    }

    /// Get the value of the "points" attribute. Convenience function.
    core::TensorList &GetPoints() { return GetPointAttr("points"); }

    /// Get the value of the "colors" attribute. Convenience function.
    core::TensorList &GetPointColors() { return GetPointAttr("colors"); }

    /// Get the value of the "normals" attribute. Convenience function.
    core::TensorList &GetPointNormals() { return GetPointAttr("normals"); }

    /// Get attributes. Throws exception if the attribute does not exist.
    ///
    /// \param key Attribute name.
    const core::TensorList &GetPointAttr(const std::string &key) const {
        return point_attr_.at(key);
    }

    /// Get the value of the "points" attribute. Convenience function.
    const core::TensorList &GetPoints() const { return GetPointAttr("points"); }

    /// Get the value of the "colors" attribute. Convenience function.
    const core::TensorList &GetPointColors() const {
        return GetPointAttr("colors");
    }

    /// Get the value of the "normals" attribute. Convenience function.
    const core::TensorList &GetPointNormals() const {
        return GetPointAttr("normals");
    }

    /// Set attributes. If the attribute key already exists, its value
    /// will be overwritten, otherwise, the new key will be created.
    ///
    /// \param key Attribute name.
    /// \param value A tensorlist.
    void SetPointAttr(const std::string &key, const core::TensorList &value) {
        point_attr_[key] = value;
    }

    /// Set the value of the "points" attribute. Convenience function.
    void SetPoints(const core::TensorList &value) {
        value.AssertElementShape({3});
        SetPointAttr("points", value);
    }

    /// Set the value of the "colors" attribute. Convenience function.
    void SetPointColors(const core::TensorList &value) {
        value.AssertElementShape({3});
        SetPointAttr("colors", value);
    }

    /// Set the value of the "normals" attribute. Convenience function.
    void SetPointNormals(const core::TensorList &value) {
        value.AssertElementShape({3});
        SetPointAttr("normals", value);
    }

    /// Returns true if all of the following is true:
    /// 1) attribute key exist
    /// 2) attribute's length as points' length
    /// 3) attribute's length > 0
    bool HasPointAttr(const std::string &key) const {
        return point_attr_.Contains(key) && GetPointAttr(key).GetSize() > 0 &&
               GetPointAttr(key).GetSize() == GetPoints().GetSize();
    }

    /// Check if the "points" attribute's value has length > 0.
    /// This is a convenience function.
    bool HasPoints() const { return HasPointAttr("points"); }

    /// Returns true if all of the following is true:
    /// 1) attribute "colors" exist
    /// 2) attribute "colors"'s length as points' length
    /// 3) attribute "colors"'s length > 0
    /// This is a convenience function.
    bool HasPointColors() const { return HasPointAttr("colors"); }

    /// Returns true if all of the following is true:
    /// 1) attribute "normals" exist
    /// 2) attribute "normals"'s length as points' length
    /// 3) attribute "normals"'s length > 0
    /// This is a convenience function.
    bool HasPointNormals() const { return HasPointAttr("normals"); }

    /// Synchronized push back, data will be copied. Before push back, all
    /// existing tensorlists must have the same length.
    ///
    /// \param map_keys_to_tensors The keys and values to be pushed back. It
    /// must contain the same keys and each corresponding tensor must have the
    /// same dtype and device.
    void SynchronizedPushBack(
            const std::unordered_map<std::string, core::Tensor>
                    &map_keys_to_tensors);

    /// Clear all data in the pointcloud.
    PointCloud &Clear() override;

    /// Returns !HasPoints().
    bool IsEmpty() const override;

    core::Tensor GetMinBound() const;

    core::Tensor GetMaxBound() const;

    core::Tensor GetCenter() const;

    PointCloud &Transform(const core::Tensor &transformation);

    PointCloud &Translate(const core::Tensor &translation,
                          bool relative = true);

    PointCloud &Scale(double scale, const core::Tensor &center);

    PointCloud &Rotate(const core::Tensor &R, const core::Tensor &center);

    /// Create a PointCloud from a legacy Open3D PointCloud.
    static tgeometry::PointCloud FromLegacyPointCloud(
            const geometry::PointCloud &pcd_legacy,
            core::Dtype dtype = core::Dtype::Float32,
            const core::Device &device = core::Device("CPU:0"));

    /// Convert to a legacy Open3D PointCloud.
    geometry::PointCloud ToLegacyPointCloud() const;

protected:
    core::Device device_ = core::Device("CPU:0");
    TensorListMap point_attr_;
};

}  // namespace tgeometry
}  // namespace open3d
