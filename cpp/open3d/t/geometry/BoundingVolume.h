// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/core/Tensor.h"
#include "open3d/core/TensorCheck.h"
#include "open3d/geometry/BoundingVolume.h"
#include "open3d/t/geometry/DrawableGeometry.h"
#include "open3d/t/geometry/Geometry.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace t {
namespace geometry {

class OrientedBoundingBox;

/// \class AxisAlignedBoundingBox
/// \brief A bounding box that is aligned along the coordinate axes and defined
/// by the min_bound and max_bound.
///
/// - (min_bound, max_bound): Lower and upper bounds of the bounding box for all
/// axes.
///     - Usage
///         - AxisAlignedBoundingBox::GetMinBound()
///         - AxisAlignedBoundingBox::SetMinBound(const core::Tensor &min_bound)
///         - AxisAlignedBoundingBox::GetMaxBound()
///         - AxisAlignedBoundingBox::SetMaxBound(const core::Tensor &max_bound)
///     - Value tensor must have shape {3,}.
///     - Value tensor must have the same data type and device.
///     - Value tensor can only be float32 (default) or float64.
///     - The device of the tensor determines the device of the box.
///
/// - color: Color of the bounding box.
///     - Usage
///         - AxisAlignedBoundingBox::GetColor()
///         - AxisAlignedBoundingBox::SetColor(const core::Tensor &color)
///     - Value tensor must have shape {3,}.
///     - Value tensor can only be float32 (default) or float64.
///     - Value tensor can only be range [0.0, 1.0].
class AxisAlignedBoundingBox : public Geometry, public DrawableGeometry {
public:
    /// \brief Construct an empty AxisAlignedBoundingBox on the provided device.
    AxisAlignedBoundingBox(const core::Device &device = core::Device("CPU:0"));

    /// \brief Construct an AxisAlignedBoundingBox from min/max bound.
    ///
    /// The AxisAlignedBoundingBox will be created on the device of the given
    /// bound tensor, which must be on the same device and have the same data
    /// type.
    /// \param min_bound Lower bounds of the bounding box for all axes. Tensor
    /// of shape {3,}, and type float32 or float64.
    /// \param max_bound Upper bounds of the bounding box for all axes. Tensor
    /// of shape {3,}, and type float32 or float64.
    AxisAlignedBoundingBox(const core::Tensor &min_bound,
                           const core::Tensor &max_bound);

    virtual ~AxisAlignedBoundingBox() override {}

    /// \brief Returns the device attribute of this AxisAlignedBoundingBox.
    core::Device GetDevice() const override { return device_; }

    /// \brief Returns the data type attribute of this AxisAlignedBoundingBox.
    core::Dtype GetDtype() const { return dtype_; }

    /// \brief Transfer the AxisAlignedBoundingBox to a specified device.
    /// \param device The targeted device to convert to.
    /// \param copy If true, a new AxisAlignedBoundingBox is always created; If
    /// false, the copy is avoided when the original AxisAlignedBoundingBox is
    /// already on the targeted device.
    AxisAlignedBoundingBox To(const core::Device &device,
                              bool copy = false) const;

    /// Returns copy of the AxisAlignedBoundingBox on the same device.
    AxisAlignedBoundingBox Clone() const {
        return To(GetDevice(), /*copy=*/true);
    }

    AxisAlignedBoundingBox &Clear() override;

    bool IsEmpty() const override { return Volume() == 0; }

    /// \brief Set the min bound of the box.
    /// If the data type of the given tensor differs from the data type of the
    /// box, an exception will be thrown.
    ///
    /// If the min bound makes the box invalid, it will not be set to the box.
    /// \param min_bound Tensor with {3,} shape, and type float32 or float64.
    void SetMinBound(const core::Tensor &min_bound);

    /// \brief Set the max boundof the box.
    /// If the data type of the given tensor differs from the data type of the
    /// box, an exception will be thrown.
    ///
    /// If the max bound makes the box invalid, it will not be set to the box.
    /// \param min_bound Tensor with {3,} shape, and type float32 or float64.
    void SetMaxBound(const core::Tensor &max_bound);

    /// \brief Set the color of the box.
    /// If the data type of the given tensor differs from the data type of the
    /// box, an exception will be thrown.
    ///
    /// \param color Tensor with {3,} shape, and type float32 or float64,
    /// with values in range [0.0, 1.0].
    void SetColor(const core::Tensor &color);

public:
    core::Tensor GetMinBound() const { return min_bound_; }

    core::Tensor GetMaxBound() const { return max_bound_; }

    core::Tensor GetColor() const { return color_; }

    core::Tensor GetCenter() const { return (min_bound_ + max_bound_) * 0.5; }

    /// \brief Translate the axis-aligned box by the given translation.
    ///
    /// If relative is true, the translation is applied to the current min and
    /// max bound. If relative is false, the translation is applied to make the
    /// box's center at the given translation.
    ///
    /// \param translation Translation tensor of shape (3,), type float32 or
    /// float64, device same as the box.
    /// \param relative Whether to perform relative translation.
    AxisAlignedBoundingBox &Translate(const core::Tensor &translation,
                                      bool relative = true);

    /// \brief Scale the axis-aligned box.
    /// If \f$mi\f$ is the min_bound and \f$ma\f$ is the max_bound of
    /// the axis aligned bounding box, and \f$s\f$ and \f$c\f$ are the
    /// provided scaling factor and center respectively, then the new
    /// min_bound and max_bound are given by \f$mi = c + s (mi - c)\f$
    /// and \f$ma = c + s (ma - c)\f$.
    /// The scaling center will be the box center if it is not specified.
    ///
    /// \param scale The scale parameter.
    /// \param center Center used for the scaling operation. Tensor of shape
    /// {3,}, type float32 or float64, device same as the box.
    AxisAlignedBoundingBox &Scale(
            double scale,
            const utility::optional<core::Tensor> &center = utility::nullopt);

    /// \brief Add operation for axis-aligned bounding box.
    /// The device of ohter box must be the same as the device of the current
    /// box.
    AxisAlignedBoundingBox &operator+=(const AxisAlignedBoundingBox &other);

    /// Get the extent/length of the bounding box in x, y, and z dimension.
    core::Tensor GetExtent() const { return max_bound_ - min_bound_; }

    /// Returns the half extent of the bounding box.
    core::Tensor GetHalfExtent() const { return GetExtent() / 2; }

    /// Returns the maximum extent, i.e. the maximum of X, Y and Z axis'
    /// extents.
    double GetMaxExtent() const {
        return GetExtent().Max({0}).To(core::Float64).Item<double>();
    }

    double GetXPercentage(double x) const;

    double GetYPercentage(double y) const;

    double GetZPercentage(double z) const;

    /// Returns the volume of the bounding box.
    double Volume() const {
        return GetExtent().Prod({0}).To(core::Float64).Item<double>();
    }

    /// Returns the eight points that define the bounding box. The Return tensor
    /// has shape {8, 3} and data type same as the box.
    core::Tensor GetBoxPoints() const;

    /// \brief Indices to points that are within the bounding box.
    ///
    /// \param points Tensor with {N, 3} shape, and type float32 or float64.
    core::Tensor GetPointIndicesWithinBoundingBox(
            const core::Tensor &points) const;

    /// Text description.
    std::string ToString() const;

    /// Convert to a legacy Open3D axis-aligned box.
    open3d::geometry::AxisAlignedBoundingBox ToLegacy() const;

    /// Convert to an oriented box.
    OrientedBoundingBox GetOrientedBoundingBox() const;

    /// Create an AxisAlignedBoundingBox from a legacy Open3D
    /// axis-aligned box.
    ///
    /// \param box Legacy AxisAlignedBoundingBox.
    /// \param dtype The data type of the box for min_bound, max_bound and
    /// color. The default is float32.
    /// \param device The device of the box. The default is CPU:0.
    static AxisAlignedBoundingBox FromLegacy(
            const open3d::geometry::AxisAlignedBoundingBox &box,
            const core::Dtype &dtype = core::Float32,
            const core::Device &device = core::Device("CPU:0"));

    /// Creates the axis-aligned box that encloses the set of points.
    /// \param points A list of points with data type of float32 or float64 (N x
    /// 3 tensor, where N must be larger than 3).
    /// \return AxisAlignedBoundingBox with same data type and device as input
    /// points.
    static AxisAlignedBoundingBox CreateFromPoints(const core::Tensor &points);

protected:
    core::Device device_ = core::Device("CPU:0");
    core::Dtype dtype_ = core::Float32;
    core::Tensor min_bound_;
    core::Tensor max_bound_;
    core::Tensor color_;
};

/// \class OrientedBoundingBox
/// \brief A bounding box oriented along an arbitrary frame of reference.
///
/// - (center, rotation, extent): The oriented bounding box is defined by its
/// center position, rotation maxtrix and extent.
///     - Usage
///         - OrientedBoundingBox::GetCenter()
///         - OrientedBoundingBox::SetCenter(const core::Tensor &center)
///         - OrientedBoundingBox::GetRotation()
///         - OrientedBoundingBox::SetRotation(const core::Tensor &rotation)
///     - Value tensor of center and extent must have shape {3,}.
///     - Value tensor of rotation must have shape {3, 3}.
///     - Value tensor must have the same data type and device.
///     - Value tensor can only be float32 (default) or float64.
///     - The device of the tensor determines the device of the box.
///
/// - color: Color of the bounding box.
///     - Usage
///         - OrientedBoundingBox::GetColor()
///         - OrientedBoundingBox::SetColor(const core::Tensor &color)
///     - Value tensor must have shape {3,}.
///     - Value tensor can only be float32 (default) or float64.
///     - Value tensor can only be range [0.0, 1.0].
class OrientedBoundingBox : public Geometry, public DrawableGeometry {
public:
    /// \brief Construct an empty OrientedBoundingBox on the provided device.
    OrientedBoundingBox(const core::Device &device = core::Device("CPU:0"));

    /// \brief Construct an OrientedBoundingBox from center, rotation and
    /// extent.
    ///
    /// The OrientedBoundingBox will be created on the device of the given
    /// tensors, which must be on the same device and have the same data
    /// type.
    /// \param center Center of the bounding box. Tensor of shape {3,}, and type
    /// float32 or float64.
    /// \param rotation Rotation matrix of the bounding box. Tensor of shape {3,
    /// 3}, and type float32 or float64.
    /// \param extent Extent of the bounding box. Tensor of shape {3,}, and type
    /// float32 or float64.
    OrientedBoundingBox(const core::Tensor &center,
                        const core::Tensor &rotation,
                        const core::Tensor &extent);

    virtual ~OrientedBoundingBox() override {}

    /// \brief Returns the device attribute of this OrientedBoundingBox.
    core::Device GetDevice() const override { return device_; }

    /// \brief Returns the data type attribute of this OrientedBoundingBox.
    core::Dtype GetDtype() const { return dtype_; }

    /// Transfer the OrientedBoundingBox to a specified device.
    /// \param device The targeted device to convert to.
    /// \param copy If true, a new OrientedBoundingBox is always created; if
    /// false, the copy is avoided when the original OrientedBoundingBox is
    /// already on the targeted device.
    OrientedBoundingBox To(const core::Device &device, bool copy = false) const;

    /// Returns copy of the OrientedBoundingBox on the same device.
    OrientedBoundingBox Clone() const { return To(GetDevice(), /*copy=*/true); }

    OrientedBoundingBox &Clear() override;

    bool IsEmpty() const override { return Volume() == 0; }

    /// \brief Set the center of the box.
    /// If the data type of the given tensor differs from the data type of the
    /// box, an exception will be thrown.
    ///
    /// \param center Tensor with {3,} shape, and type float32 or float64.
    void SetCenter(const core::Tensor &center);

    /// \brief Set the rotation matrix of the box.
    /// If the data type of the given tensor differs from the data type of the
    /// box, an exception will be thrown.
    ///
    /// \param rotation Tensor with {3, 3} shape, and type float32 or float64.
    void SetRotation(const core::Tensor &rotation);

    /// \brief Set the extent of the box.
    /// If the data type of the given tensor differs from the data type of the
    /// box, an exception will be thrown.
    ///
    /// \param extent Tensor with {3,} shape, and type float32 or float64.
    void SetExtent(const core::Tensor &extent);

    /// \brief Set the color of the box.
    ///
    /// \param color Tensor with {3,} shape, and type float32 or float64,
    /// with values in range [0.0, 1.0].
    void SetColor(const core::Tensor &color);

public:
    core::Tensor GetMinBound() const;

    core::Tensor GetMaxBound() const;

    core::Tensor GetColor() const { return color_; }

    core::Tensor GetCenter() const { return center_; }

    core::Tensor GetRotation() const { return rotation_; }

    core::Tensor GetExtent() const { return extent_; }

    /// \brief Translate the oriented box by the given translation.
    /// If relative is true, the translation is added to the center of the box.
    /// If false, the center will be assigned to the translation.
    ///
    /// \param translation Translation tensor of shape {3,}, type float32 or
    /// float64, device same as the box.
    /// \param relative Whether to perform relative translation.
    OrientedBoundingBox &Translate(const core::Tensor &translation,
                                   bool relative = true);

    /// \brief Rotate the oriented box by the given rotation matrix. If the
    /// rotation matrix is not orthogonal, the rotation will no be applied.
    /// The rotation center will be the box center if it is not specified.
    ///
    /// \param rotation Rotation matrix of shape {3, 3}, type float32 or
    /// float64, device same as the box.
    /// \param center Center of the rotation, default is null, which means use
    /// center of the box as rotation center.
    OrientedBoundingBox &Rotate(
            const core::Tensor &rotation,
            const utility::optional<core::Tensor> &center = utility::nullopt);

    /// \brief Transform the oriented box by the given transformation matrix.
    ///
    /// \param transformation Transformation matrix of shape {4, 4}, type
    /// float32 or float64, device same as the box.
    OrientedBoundingBox &Transform(const core::Tensor &transformation);

    /// \brief Scale the axis-aligned box.
    /// If \f$mi\f$ is the min_bound and \f$ma\f$ is the max_bound of
    /// the axis aligned bounding box, and \f$s\f$ and \f$c\f$ are the
    /// provided scaling factor and center respectively, then the new
    /// min_bound and max_bound are given by \f$mi = c + s (mi - c)\f$
    /// and \f$ma = c + s (ma - c)\f$.
    /// The scaling center will be the box center if it is not specified.
    ///
    /// \param scale The scale parameter.
    /// \param center Center used for the scaling operation. Tensor of shape
    /// {3,}, type float32 or float64, device same as the box.
    OrientedBoundingBox &Scale(
            double scale,
            const utility::optional<core::Tensor> &center = utility::nullopt);

    /// Returns the volume of the bounding box.
    double Volume() const {
        return GetExtent().Prod({0}).To(core::Float64).Item<double>();
    }

    /// Returns the eight points that define the bounding box. The Return tensor
    /// has shape {8, 3} and data type same as the box.
    ///
    /// \verbatim
    ///      ------- x
    ///     /|
    ///    / |
    ///   /  | z
    ///  y
    ///      0 ------------------- 1
    ///       /|                /|
    ///      / |               / |
    ///     /  |              /  |
    ///    /   |             /   |
    /// 2 ------------------- 7  |
    ///   |    |____________|____| 6
    ///   |   /3            |   /
    ///   |  /              |  /
    ///   | /               | /
    ///   |/                |/
    /// 5 ------------------- 4
    /// \endverbatim
    core::Tensor GetBoxPoints() const;

    /// \brief Indices to points that are within the bounding box.
    ///
    /// \param points Tensor with {N, 3} shape, and type float32 or float64.
    core::Tensor GetPointIndicesWithinBoundingBox(
            const core::Tensor &points) const;

    /// Text description.
    std::string ToString() const;

    /// Convert to a legacy Open3D oriented box.
    open3d::geometry::OrientedBoundingBox ToLegacy() const;

    /// Convert to an axis-aligned box.
    AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const;

    /// Create an oriented bounding box from the AxisAlignedBoundingBox.
    ///
    /// \param aabb AxisAlignedBoundingBox object from which
    /// OrientedBoundingBox is created.
    /// \return OrientedBoundingBox with the same device and dtype as input box.
    static OrientedBoundingBox CreateFromAxisAlignedBoundingBox(
            const AxisAlignedBoundingBox &aabb);

    /// Create an OrientedBoundingBox from a legacy Open3D oriented box.
    ///
    /// \param box Legacy OrientedBoundingBox.
    /// \param dtype The data type of the box for min_bound max_bound and color.
    /// The default is float32.
    /// \param device The device of the box. The default is CPU:0.
    static OrientedBoundingBox FromLegacy(
            const open3d::geometry::OrientedBoundingBox &box,
            const core::Dtype &dtype = core::Float32,
            const core::Device &device = core::Device("CPU:0"));

    /// Creates an oriented bounding box using a PCA.
    /// Note that this is only an approximation to the minimum oriented
    /// bounding box that could be computed for example with O'Rourke's
    /// algorithm (cf. http://cs.smith.edu/~jorourke/Papers/MinVolBox.pdf,
    /// https://www.geometrictools.com/Documentation/MinimumVolumeBox.pdf)
    /// This is a wrapper for a CPU implementation.
    ///
    /// \param points A list of points with data type of float32 or float64 (N x
    /// 3 tensor, where N must be larger than 3).
    /// \param robust If set to true uses a more robust method which works in
    /// degenerate cases but introduces noise to the points coordinates.
    /// \return OrientedBoundingBox with same data type and device as input
    /// points.
    static OrientedBoundingBox CreateFromPoints(const core::Tensor &points,
                                                bool robust = false);

protected:
    core::Device device_ = core::Device("CPU:0");
    core::Dtype dtype_ = core::Float32;
    core::Tensor center_;
    core::Tensor rotation_;
    core::Tensor extent_;
    core::Tensor color_;
};

}  // namespace geometry
}  // namespace t
}  // namespace open3d
