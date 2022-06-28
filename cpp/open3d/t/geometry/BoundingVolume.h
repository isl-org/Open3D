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
///
/// \brief A bounding box that is aligned along the coordinate axes.
///
///  The AxisAlignedBoundingBox uses the coordinate axes for bounding box
///  generation. This means that the bounding box is oriented along the
///  coordinate axes.
class AxisAlignedBoundingBox : public Geometry, public DrawableGeometry {
public:
    /// \brief Construct an empty AxisAlignedBoundingBox on the provided device.
    AxisAlignedBoundingBox(const core::Device &device = core::Device("CPU:0"));

    /// \brief Construct an AxisAlignedBoundingBox from min/max bound.
    ///
    /// The AxisAlignedBoundingBox will be created on the device of the given
    /// bound tensor, which must be on the same device.
    ///
    /// \param min_bound Lower bounds of the bounding box for all axes.
    /// \param max_bound Upper bounds of the bounding box for all axes.
    AxisAlignedBoundingBox(const core::Tensor &min_bound,
                           const core::Tensor &max_bound);

    virtual ~AxisAlignedBoundingBox() override {}

    /// \brief Returns the device attribute of this AxisAlignedBoundingBox.
    core::Device GetDevice() const override { return device_; }

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

    bool IsEmpty() const override { return Volume() <= 0; }

    void SetMinBound(const core::Tensor &min_bound) {
        core::AssertTensorDevice(min_bound, device_);
        core::AssertTensorShape(min_bound, {3});
        min_bound_ = min_bound;
    }

    void SetMaxBound(const core::Tensor &max_bound) {
        core::AssertTensorDevice(max_bound, device_);
        core::AssertTensorShape(max_bound, {3});
        max_bound_ = max_bound;
    }

    void SetColor(const core::Tensor &color) {
        core::AssertTensorDevice(color, device_);
        core::AssertTensorShape(color, {3});
        color_ = color;
    }

    core::Tensor &GetMinBound() { return min_bound_; }
    core::Tensor &GetMaxBound() { return max_bound_; }
    core::Tensor &GetColor() { return color_; }

    const core::Tensor &GetMinBound() const { return min_bound_; }
    const core::Tensor &GetMaxBound() const { return max_bound_; }
    const core::Tensor &GetColor() const { return color_; }
    core::Tensor GetCenter() const { return (min_bound_ + max_bound_) * 0.5; }

    AxisAlignedBoundingBox &Translate(const core::Tensor &translation,
                                      bool relative = true);

    /// \brief Scales the axis-aligned bounding boxes.
    /// If \f$mi\f$ is the min_bound and \f$ma\f$ is the max_bound of
    /// the axis aligned bounding box, and \f$s\f$ and \f$c\f$ are the
    /// provided scaling factor and center respectively, then the new
    /// min_bound and max_bound are given by \f$mi = c + s (mi - c)\f$
    /// and \f$ma = c + s (ma - c)\f$.
    ///
    /// \param scale The scale parameter.
    /// \param center Center used for the scaling operation.
    AxisAlignedBoundingBox &Scale(double scale, const core::Tensor &center);

    AxisAlignedBoundingBox &operator+=(const AxisAlignedBoundingBox &other);

    /// Get the extent/length of the bounding box in x, y, and z dimension.
    core::Tensor GetExtent() const { return max_bound_ - min_bound_; }

    /// Returns the half extent of the bounding box.
    core::Tensor GetHalfExtent() const { return GetExtent() / 2; }

    /// Returns the maximum extent, i.e. the maximum of X, Y and Z axis'
    /// extents.
    double GetMaxExtent() const { return GetExtent().Max({0}).Item<double>(); }

    double GetXPercentage(double x) const;
    double GetYPercentage(double y) const;
    double GetZPercentage(double z) const;

    /// Returns the volume of the bounding box.
    double Volume() const { return GetExtent().Prod({0}).Item<double>(); }

    /// Returns the eight points that define the bounding box.
    core::Tensor GetBoxPoints() const;

    /// Return indices to points that are within the bounding box.
    ///
    /// \param points A list of points A list of points (N x 3 tensor).
    core::Tensor GetPointIndicesWithinBoundingBox(
            const core::Tensor &points) const;

    /// Returns the 3D dimensions of the bounding box in string format.
    std::string GetPrintInfo() const;

    /// Convert to a legacy Open3D AxisAlignedBoundingBox.
    open3d::geometry::AxisAlignedBoundingBox ToLegacy() const;

    /// Create an AxisAlignedBoundingBox from a legacy Open3D
    /// AxisAlignedBoundingBox.
    static AxisAlignedBoundingBox FromLegacy(
            const open3d::geometry::AxisAlignedBoundingBox &box,
            core::Dtype dtype = core::Float32,
            const core::Device &device = core::Device("CPU:0"));

    /// Creates the AxisAlignedBoundingBox that encloses the set of points.
    ///
    /// \param points A list of points (N x 3 tensor, where N must be larger
    /// than 3).
    static AxisAlignedBoundingBox CreateFromPoints(const core::Tensor &points);

protected:
    core::Device device_ = core::Device("CPU:0");
    core::Tensor min_bound_;
    core::Tensor max_bound_;
    core::Tensor color_;
};

}  // namespace geometry
}  // namespace t
}  // namespace open3d
