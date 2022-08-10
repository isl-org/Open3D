// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "open3d/core/Tensor.h"
#include "open3d/core/TensorCheck.h"
#include "open3d/geometry/BoundingVolume.h"
#include "open3d/t/geometry/DrawableGeometry.h"
#include "open3d/t/geometry/Geometry.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace t {
namespace geometry {

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

    /// Transfer the AxisAlignedBoundingBox to a specified device.
    /// \param device The targeted device to convert to.
    /// \param copy If true, a new AxisAlignedBoundingBox is always created; if
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
    /// original tensor, it will be converted into the same data type.
    /// If the min bound makes the box invalid, it will not be set to the box.
    /// \param min_bound Tensor with {3,} shape, and type float32 or float64.
    void SetMinBound(const core::Tensor &min_bound);

    /// \brief Set the max boundof the box.
    /// If the data type of the given tensor differs from the data type of the
    /// original tensor, it will be converted into the same data type.
    /// If the max bound makes the box invalid, it will not be set to the box.
    /// \param min_bound Tensor with {3,} shape, and type float32 or float64.
    void SetMaxBound(const core::Tensor &max_bound);

    /// \brief Set the color of the box.
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
    ///
    /// \param scale The scale parameter.
    /// \param center Center used for the scaling operation. Tensor of shape
    /// {3,}, type float32 or float64, device same as the box.
    AxisAlignedBoundingBox &Scale(double scale, const core::Tensor &center);

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
    /// has shape {8, 3} and data type of float32.
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

    /// Create an AxisAlignedBoundingBox from a legacy Open3D
    /// axis-aligned box.
    ///
    /// \param dtype The data type of the box for min_bound max_bound and color.
    /// The default is float32. \param device The device of the box. The default
    /// is CPU:0.
    static AxisAlignedBoundingBox FromLegacy(
            const open3d::geometry::AxisAlignedBoundingBox &box,
            const core::Dtype &dtype = core::Float32,
            const core::Device &device = core::Device("CPU:0"));

    /// Creates the axis-aligned box that encloses the set of points.
    /// \param points A list of points with data type of float32 or float64 (N x
    /// 3 tensor, where N must be larger than 3).
    static AxisAlignedBoundingBox CreateFromPoints(const core::Tensor &points);

protected:
    core::Device device_ = core::Device("CPU:0");
    core::Dtype dtype_ = core::Float32;
    core::Tensor min_bound_;
    core::Tensor max_bound_;
    core::Tensor color_;
};

}  // namespace geometry
}  // namespace t
}  // namespace open3d
