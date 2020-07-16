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
#include "open3d/tgeometry/Geometry3D.h"

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
/// attributes has different levels:
///
/// - Level 0: Default attribute {"points"}.
///     - Level 0 attribute "points" is created by default.
///     - Level 0 attribute is present in all pointclouds and it is enforced
///       programatically.
///     - Level 0 attribute has convenience functions. Examples:
///         - PointCloud::GetPoints()
///         - PointCloud::SetPoints(points_tensorlist)
///         - PointCloud::HasPoints()
/// - Level 1: Commonly-used attributes {"normals", "colors"}.
///     - Level 1 attributes are not created by default.
///     - Level 1 function has convenience functions. Note the naming convention
///     is different from level 0 attributes. Examples:
///         - PointCloud::GetPointNormals()
///         - PointCloud::SetPointNormals(normals_tensorlist)
///         - PointCloud::HasPointNormals()
///         - PointCloud::GetPointColors()
///         - PointCloud::SetPointColors(colors_tensorlist)
///         - PointCloud::HasPointColors()
/// - Level 2: Custom attributes, e.g. {"labels", "alphas", "intensities"}.
///     - Level 2 attributes are created by users.
///     - Generalized helper functions are provided. Examples:
///         - PointCloud::GetPointAttr("labels")
///         - PointCloud::SetPointAttr("labels", labels_tensorlist)
///         - PointCloud::HasPointAttr("labels")
///     - Note that the level 0 and level 1 convenience functions can also be
///     achieved via the generalized helper functions:
///         - PointCloud::GetPoints() is the same as
///           PointCloud::GetPointAttr("points")
///         - PointCloud::HasPointNormals() is the same as
///           PointCloud::HasPointAttr("normals")
class PointCloud : public Geometry3D {
public:
    /// Construct an empty pointcloud.
    PointCloud(core::Dtype dtype = core::Dtype::Float32,
               const core::Device &device = core::Device("CPU:0"))
        : Geometry3D(Geometry::GeometryType::PointCloud),
          dtype_(dtype),
          device_(device) {
        point_attr_["points"] = core::TensorList({3}, dtype_, device_);
    }

    /// Construct a pointcloud from points.
    ///
    /// \param points A tensorlist with element shape (3,).
    /// - The resulting pointcloud will have the same dtype and device as the
    /// tensorlist.
    /// - If the tensorlist is created in-place from a pre-allocated buffer, the
    /// tensorlist has a fixed size and thus the resulting pointcloud will have
    /// a fixed size and calling to functions like `SyncPushBack` will raise an
    /// exception.
    PointCloud(const core::TensorList &points);

    /// Construct from points and other attributes of the points.
    ///
    /// \param point_attr A map of string to TensorList containing points and
    /// their attributes. point_dict must contain at least the "points" key.
    PointCloud(const std::unordered_map<std::string, core::TensorList>
                       &point_attr);

    virtual ~PointCloud() override {}

    /// Get point attributes. Throws exception if the attribute name does not
    /// exist.
    ///
    /// \param key Attribute name. Typical attribute name includes "points",
    /// "colors", "normals".
    core::TensorList &GetPointAttr(const std::string &key);

    const core::TensorList &GetConstAttr(const std::string &key) const {
        return point_attr_.at(key);
    }

    /// Set point attributes. If the attribute key already exists, its value
    /// will be overwritten, otherwise, the new key will be created.
    ///
    /// \param key Attribute name. Typical attribute name includes "points",
    /// "colors", "normals".
    /// \param value A tensorlist.
    void SetPointAttr(const std::string &key, const core::TensorList &value);

    core::TensorList &operator[](const std::string &key);

    /// Synchronized push back. Before push back, the function asserts that all
    /// the tensorlists in the point_attr_ have the same length.
    ///
    /// \param point_struct The keys and values to be pushed back to
    /// PointCloud::point_attr_. \p point_struct 's keys must be exactly the
    /// same as PointCloud::point_attr_'s keys and the each of the corresponding
    /// tensors must have the same shape, dtype, devie as the ones stored in
    /// point_attr_.
    void SyncPushBack(
            const std::unordered_map<std::string, core::Tensor> &point_struct);

    /// Clear all data in the pointcloud.
    PointCloud &Clear() override;

    core::Tensor GetMinBound() const override;

    core::Tensor GetMaxBound() const override;

    core::Tensor GetCenter() const override;

    PointCloud &Transform(const core::Tensor &transformation) override;

    PointCloud &Translate(const core::Tensor &translation,
                          bool relative = true) override;

    PointCloud &Scale(double scale, const core::Tensor &center) override;

    PointCloud &Rotate(const core::Tensor &R,
                       const core::Tensor &center) override;

    /// Returns !HasPoints().
    bool IsEmpty() const override;

    /// Returns true if the pointcloud contains 1 or more points.
    bool HasPoints() const { return point_attr_.at("points").GetSize() > 0; }

    /// Returns true if the pointcloud contains 1 or more colors for points.
    bool HasColors() const {
        return HasPoints() && point_attr_.count("colors") != 0 &&
               point_attr_.at("colors").GetSize() ==
                       point_attr_.at("points").GetSize();
    }

    /// Returns true if the pointcloud contains 1 or more normals for points.
    bool HasNormals() const {
        return HasPoints() && point_attr_.count("normals") != 0 &&
               point_attr_.at("normals").GetSize() ==
                       point_attr_.at("points").GetSize();
    }

    /// Create a PointCloud from a legacy Open3D PointCloud.
    static tgeometry::PointCloud FromLegacyPointCloud(
            const geometry::PointCloud &pcd_legacy,
            core::Dtype dtype = core::Dtype::Float32,
            const core::Device &device = core::Device("CPU:0"));

    /// Convert to a legacy Open3D PointCloud.
    geometry::PointCloud ToLegacyPointCloud() const;

protected:
    std::unordered_map<std::string, core::TensorList> point_attr_;
    core::Dtype dtype_ = core::Dtype::Float32;
    core::Device device_ = core::Device("CPU:0");
};

}  // namespace tgeometry
}  // namespace open3d
