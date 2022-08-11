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

#include "open3d/t/geometry/BoundingVolume.h"

#include "open3d/core/EigenConverter.h"
#include "open3d/core/TensorFunction.h"
#include "open3d/t/geometry/kernel/PointCloud.h"

namespace open3d {
namespace t {
namespace geometry {

AxisAlignedBoundingBox::AxisAlignedBoundingBox(const core::Device &device)
    : Geometry(Geometry::GeometryType::AxisAlignedBoundingBox, 3),
      device_(device),
      dtype_(core::Float32),
      min_bound_(core::Tensor::Zeros({3}, dtype_, device)),
      max_bound_(core::Tensor::Zeros({3}, dtype_, device)),
      color_(core::Tensor::Ones({3}, dtype_, device)) {}

AxisAlignedBoundingBox::AxisAlignedBoundingBox(const core::Tensor &min_bound,
                                               const core::Tensor &max_bound)
    : AxisAlignedBoundingBox([&]() {
          core::AssertTensorDevice(max_bound, min_bound.GetDevice());
          core::AssertTensorDtype(max_bound, min_bound.GetDtype());
          core::AssertTensorShape(min_bound, {3});
          core::AssertTensorShape(max_bound, {3});
          return min_bound.GetDevice();
      }()) {
    device_ = min_bound.GetDevice();
    dtype_ = min_bound.GetDtype();

    min_bound_ = min_bound;
    max_bound_ = max_bound;
    color_ = core::Tensor::Ones({3}, dtype_, device_);

    // Check if the bounding box is valid.
    if (Volume() < 0) {
        utility::LogError(
                "Invalid axis-aligned bounding box. Please make sure all "
                "the elements in max bound are larger than min bound.");
    }
}

AxisAlignedBoundingBox AxisAlignedBoundingBox::To(const core::Device &device,
                                                  bool copy) const {
    if (!copy && GetDevice() == device) {
        return *this;
    }
    AxisAlignedBoundingBox box(device);
    box.SetMaxBound(max_bound_.To(device, true));
    box.SetMinBound(min_bound_.To(device, true));
    box.SetColor(color_.To(device, true));
    return box;
}

AxisAlignedBoundingBox &AxisAlignedBoundingBox::Clear() {
    min_bound_ = core::Tensor::Zeros({3}, GetDtype(), GetDevice());
    max_bound_ = core::Tensor::Zeros({3}, GetDtype(), GetDevice());
    color_ = core::Tensor::Ones({3}, GetDtype(), GetDevice());
    return *this;
}

void AxisAlignedBoundingBox::SetMinBound(const core::Tensor &min_bound) {
    core::AssertTensorDevice(min_bound, device_);
    core::AssertTensorShape(min_bound, {3});
    const core::Tensor tmp = min_bound_.Clone();
    min_bound_ = min_bound.To(min_bound_.GetDtype());

    // If the volume is invalid, the min_bound_ will be set to the
    // original value.
    if (Volume() < 0) {
        utility::LogWarning(
                "Invalid axis-aligned bounding box. Please make sure all "
                "the elements in min bound are smaller than min bound.");
        min_bound_ = tmp;
    }
}

void AxisAlignedBoundingBox::SetMaxBound(const core::Tensor &max_bound) {
    core::AssertTensorDevice(max_bound, device_);
    core::AssertTensorShape(max_bound, {3});
    core::AssertTensorDtype(max_bound, GetDtype());

    const core::Tensor tmp = max_bound_.Clone();
    max_bound_ = max_bound;

    // If the volume is invalid, the max_bound_ will be set to the
    // original value.
    if (Volume() < 0) {
        utility::LogWarning(
                "Invalid axis-aligned bounding box. Please make sure all "
                "the elements in max bound are larger than min bound.");
        max_bound_ = tmp;
    }
}

void AxisAlignedBoundingBox::SetColor(const core::Tensor &color) {
    core::AssertTensorDevice(color, GetDevice());
    core::AssertTensorShape(color, {3});
    core::AssertTensorDtype(color, GetDtype());
    if (color.Max({0}).To(core::Float64).Item<double>() > 1.0 ||
        color.Min({0}).To(core::Float64).Item<double>() < 0.0) {
        utility::LogError(
                "The color must be in the range [0, 1], but for range [{}, "
                "{}].",
                color.Min({0}).To(core::Float64).Item<double>(),
                color.Max({0}).To(core::Float64).Item<double>());
    }

    color_ = color;
}

AxisAlignedBoundingBox &AxisAlignedBoundingBox::Translate(
        const core::Tensor &translation, bool relative) {
    core::AssertTensorDevice(translation, GetDevice());
    core::AssertTensorShape(translation, {3});
    core::AssertTensorDtype(translation, GetDtype());

    if (relative) {
        min_bound_ += translation;
        max_bound_ += translation;
    } else {
        const core::Tensor half_extent = GetHalfExtent();
        min_bound_ = translation - half_extent;
        max_bound_ = translation + half_extent;
    }
    return *this;
}

AxisAlignedBoundingBox &AxisAlignedBoundingBox::Scale(
        double scale, const core::Tensor &center) {
    core::AssertTensorDevice(center, GetDevice());
    core::AssertTensorShape(center, {3});
    core::AssertTensorDtype(center, min_bound_.GetDtype());

    min_bound_ = center + scale * (min_bound_ - center);
    max_bound_ = center + scale * (max_bound_ - center);

    return *this;
}

AxisAlignedBoundingBox &AxisAlignedBoundingBox::operator+=(
        const AxisAlignedBoundingBox &other) {
    if (other.GetDtype() != GetDtype()) {
        utility::LogError(
                "The data-type of the other bounding box is {}, but the "
                "data-type of this bounding box is {}.",
                other.GetDtype().ToString(), GetDtype().ToString());
    }

    if (IsEmpty()) {
        min_bound_ = other.GetMinBound();
        max_bound_ = other.GetMaxBound();
    } else if (!other.IsEmpty()) {
        min_bound_ = core::Minimum(min_bound_, other.GetMinBound());
        max_bound_ = core::Maximum(max_bound_, other.GetMaxBound());
    }
    return *this;
}

double AxisAlignedBoundingBox::GetXPercentage(double x) const {
    const double x_min = min_bound_[0].To(core::Float64).Item<double>();
    const double x_max = max_bound_[0].To(core::Float64).Item<double>();
    return (x - x_min) / (x_max - x_min);
}

double AxisAlignedBoundingBox::GetYPercentage(double y) const {
    const double y_min = min_bound_[1].To(core::Float64).Item<double>();
    const double y_max = max_bound_[1].To(core::Float64).Item<double>();
    return (y - y_min) / (y_max - y_min);
}

double AxisAlignedBoundingBox::GetZPercentage(double z) const {
    const double z_min = min_bound_[2].To(core::Float64).Item<double>();
    const double z_max = max_bound_[2].To(core::Float64).Item<double>();
    return (z - z_min) / (z_max - z_min);
}

core::Tensor AxisAlignedBoundingBox::GetBoxPoints() const {
    const core::Tensor extent_3x3 =
            core::Tensor::Eye(3, GetDtype(), GetDevice())
                    .Mul(GetExtent().Reshape({3, 1}));

    return core::Concatenate({min_bound_.Reshape({1, 3}),
                              min_bound_.Reshape({1, 3}) + extent_3x3,
                              max_bound_.Reshape({1, 3}),
                              max_bound_.Reshape({1, 3}) - extent_3x3});
}

core::Tensor AxisAlignedBoundingBox::GetPointIndicesWithinBoundingBox(
        const core::Tensor &points) const {
    core::AssertTensorDevice(points, GetDevice());
    core::AssertTensorShape(points, {utility::nullopt, 3});

    core::Tensor mask =
            core::Tensor::Zeros({points.GetLength()}, core::Bool, GetDevice());
    kernel::pointcloud::GetPointMaskWithinAABB(points, min_bound_, max_bound_,
                                               mask);

    return mask.NonZero().Flatten();
}

std::string AxisAlignedBoundingBox::ToString() const {
    return fmt::format("AxisAlignedBoundingBox[{}, {}]", GetDtype().ToString(),
                       GetDevice().ToString());
}

AxisAlignedBoundingBox AxisAlignedBoundingBox::CreateFromPoints(
        const core::Tensor &points) {
    core::AssertTensorShape(points, {utility::nullopt, 3});
    if (points.GetLength() <= 3) {
        utility::LogWarning("The points number is less than 3.");
        return AxisAlignedBoundingBox(points.GetDevice());
    } else {
        const core::Tensor min_bound = points.Min({0});
        const core::Tensor max_bound = points.Max({0});
        return AxisAlignedBoundingBox(min_bound.To(core::Float32),
                                      max_bound.To(core::Float32));
    }
}

open3d::geometry::AxisAlignedBoundingBox AxisAlignedBoundingBox::ToLegacy()
        const {
    open3d::geometry::AxisAlignedBoundingBox legacy_box;

    AxisAlignedBoundingBox box_new;

    // Make sure the box is in CPU.
    box_new = To(core::Device("CPU:0"));

    // TODO: The helper function for conversion between 1-D Tensor and
    // Eigen::VectorXd could be implemented in `core/EigenConverter.cpp`.
    legacy_box.min_bound_ = core::eigen_converter::TensorToEigenVector3dVector(
            box_new.GetMinBound().Reshape({1, 3}))[0];
    legacy_box.max_bound_ = core::eigen_converter::TensorToEigenVector3dVector(
            max_bound_.Reshape({1, 3}))[0];
    legacy_box.color_ = core::eigen_converter::TensorToEigenVector3dVector(
            color_.Reshape({1, 3}))[0];
    return legacy_box;
}

AxisAlignedBoundingBox AxisAlignedBoundingBox::FromLegacy(
        const open3d::geometry::AxisAlignedBoundingBox &box,
        const core::Dtype &dtype,
        const core::Device &device) {
    if (dtype != core::Float32 && dtype != core::Float64) {
        utility::LogError(
                "Got data-type {}, but the supported data-type of the bounding "
                "box are Float32 and Float64.",
                dtype.ToString());
    }

    AxisAlignedBoundingBox t_box(
            core::eigen_converter::EigenMatrixToTensor(box.min_bound_)
                    .Flatten()
                    .To(device, dtype),
            core::eigen_converter::EigenMatrixToTensor(box.max_bound_)
                    .Flatten()
                    .To(device, dtype));

    t_box.SetColor(core::eigen_converter::EigenMatrixToTensor(box.color_)
                           .Flatten()
                           .To(device, dtype));
    return t_box;
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
