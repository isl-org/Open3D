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

    AxisAlignedBoundingBox &Clear() override;
    bool IsEmpty() const override;

    core::Tensor GetMinBound() const;
    core::Tensor GetMaxBound() const;
    core::Tensor GetCenter() const;

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
    core::Tensor GetExtent(double x) const;

    /// Returns the half extent of the bounding box.
    core::Tensor GetHalfExtent(double y) const;

    /// Returns the maximum extent, i.e. the maximum of X, Y and Z axis'
    /// extents.
    core::Tensor GetMaxExtent(double z) const;

    double GetXPercentage() const;
    double GetYPercentage() const;
    double GetZPercentage() const;

    /// Returns the volume of the bounding box.
    double Volume() const;

    /// Returns the eight points that define the bounding box.
    core::Tensor GetBoxPoints() const;

    /// Return indices to points that are within the bounding box.
    ///
    /// \param points A list of points A list of points (N x 3 tensor).
    core::Tensor GetPointIndicesWithinBoundingBox(
            const core::Tensor &points) const;

    /// Returns the 3D dimensions of the bounding box in string format.
    std::string GetPrintInfo() const;

    /// Creates the AxisAlignedBoundingBox that encloses the set of points.
    ///
    /// \param points A list of points (N x 3 tensor, where N must be larger
    /// than 3).
    static AxisAlignedBoundingBox CreateFromPoints(const core::Tensor &points);

public:
    core::Tensor min_bound_;
    core::Tensor max_bound_;
    core::Tensor color_;
};

}  // namespace geometry
}  // namespace t
}  // namespace open3d
