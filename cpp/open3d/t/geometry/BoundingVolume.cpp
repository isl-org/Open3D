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
      min_bound_(core::Tensor::Zeros({3}, core::Float32, device)),
      max_bound_(core::Tensor::Zeros({3}, core::Float32, device)),
      color_(core::Tensor::Ones({3}, core::Float32, device)) {}

AxisAlignedBoundingBox::AxisAlignedBoundingBox(const core::Tensor &min_bound,
                                               const core::Tensor &max_bound)
    : AxisAlignedBoundingBox([&]() {
          core::AssertTensorDevice(min_bound, max_bound.GetDevice());
          core::AssertTensorShape(min_bound, {3});
          core::AssertTensorShape(max_bound, {3});
          return min_bound.GetDevice();
      }()) {
    if (min_bound.GetDtype() == core::Bool) {
        utility::LogError("min_bound and max_bound can not be bool.");
    }

    min_bound_ = min_bound.To(core::Float32);
    max_bound_ = max_bound.To(core::Float32);

    // Check if the bounding box is valid.
    CheckValid(true);
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
    min_bound_ = core::Tensor::Zeros({3}, core::Float32, GetDevice());
    max_bound_ = core::Tensor::Zeros({3}, core::Float32, GetDevice());
    color_ = core::Tensor::Ones({3}, core::Float32, GetDevice());
    return *this;
}

void AxisAlignedBoundingBox::SetMinBound(const core::Tensor &min_bound) {
    core::AssertTensorDevice(min_bound, device_);
    core::AssertTensorShape(min_bound, {3});
    const core::Tensor tmp = min_bound.Clone();
    min_bound_ = min_bound.To(min_bound_.GetDtype());

    // If fail to pass the valid check, the min_bound_ will be set to the
    // original value.
    if (!CheckValid(false)) {
        min_bound_ = tmp;
    }
}

void AxisAlignedBoundingBox::SetMaxBound(const core::Tensor &max_bound) {
    core::AssertTensorDevice(max_bound, device_);
    core::AssertTensorShape(max_bound, {3});
    const core::Tensor tmp = max_bound.Clone();
    max_bound_ = max_bound.To(max_bound_.GetDtype());

    // If fail to pass the valid check, the max_bound_ will be set to the
    // original value.
    if (!CheckValid(false)) {
        max_bound_ = tmp;
    }
}

void AxisAlignedBoundingBox::SetColor(const core::Tensor &color) {
    core::AssertTensorDevice(color, device_);
    core::AssertTensorShape(color, {3});
    color_ = color.To(color_.GetDevice());
}

AxisAlignedBoundingBox &AxisAlignedBoundingBox::Translate(
        const core::Tensor &translation, bool relative) {
    core::AssertTensorDevice(translation, GetDevice());
    core::AssertTensorShape(translation, {3});
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
    min_bound_ = center + scale * (min_bound_ - center);
    max_bound_ = center + scale * (max_bound_ - center);
    return *this;
}

AxisAlignedBoundingBox &AxisAlignedBoundingBox::operator+=(
        const AxisAlignedBoundingBox &other) {
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
    const core::Tensor extent = GetExtent().To(core::Float32);
    core::Tensor points =
            core::Tensor::Zeros({8, 3}, core::Float32, GetDevice());
    const float *extent_data = extent.GetDataPtr<float>();

    points[0] = min_bound_;
    points[1] = min_bound_ +
                core::Tensor::Init<float>({extent_data[0], 0, 0}, GetDevice());
    points[2] = min_bound_ +
                core::Tensor::Init<float>({0, extent_data[1], 0}, GetDevice());
    points[3] = min_bound_ +
                core::Tensor::Init<float>({0, 0, extent_data[2]}, GetDevice());
    points[4] = max_bound_;
    points[5] = max_bound_ +
                core::Tensor::Init<float>({-extent_data[0], 0, 0}, GetDevice());
    points[6] = max_bound_ +
                core::Tensor::Init<float>({0, -extent_data[1], 0}, GetDevice());
    points[7] = max_bound_ +
                core::Tensor::Init<float>({0, 0, -extent_data[2]}, GetDevice());
    return points;
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
    const core::Dtype dtype = min_bound_.GetDtype();
    return fmt::format("AxisAlignedBoundingBox[{}, {}]", dtype.ToString(),
                       device_.ToString());
}

AxisAlignedBoundingBox AxisAlignedBoundingBox::CreateFromPoints(
        const core::Tensor &points) {
    core::AssertTensorShape(points, {utility::nullopt, 3});
    core::AssertTensorDtypes(points, {core::Float32, core::Float64});
    if (points.GetLength() <= 0) {
        utility::LogWarning("The points number is less than 3.");
        return AxisAlignedBoundingBox(points.GetDevice());
    } else {
        const core::Tensor min_bound = points.Min({1});
        const core::Tensor max_bound = points.Max({1});
        return AxisAlignedBoundingBox(min_bound, max_bound);
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
        core::Dtype dtype,
        const core::Device &device) {
    AxisAlignedBoundingBox t_box(device);
    t_box.SetColor(core::eigen_converter::EigenMatrixToTensor(box.color_)
                           .Flatten()
                           .To(device));
    t_box.SetMinBound(core::eigen_converter::EigenMatrixToTensor(box.min_bound_)
                              .Flatten()
                              .To(device));
    t_box.SetMaxBound(core::eigen_converter::EigenMatrixToTensor(box.max_bound_)
                              .Flatten()
                              .To(device));
    return t_box;
}

bool AxisAlignedBoundingBox::CheckValid(bool exception) const {
    if (Volume() < 0) {
        if (exception) {
            utility::LogError(
                    "Invalid axis-aligned bounding box. Please make sure all "
                    "the elements in max bound are larger than min bound.");
        } else {
            utility::LogWarning(
                    "Invalid axis-aligned bounding box. Please make sure all "
                    "the elements in max bound are larger than min bound.");
            return false;
        }
    }
    return true;
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
