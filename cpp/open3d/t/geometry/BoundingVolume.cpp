// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
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
          core::AssertTensorDtypes(max_bound, {core::Float32, core::Float64});
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
    core::AssertTensorDevice(min_bound, GetDevice());
    core::AssertTensorShape(min_bound, {3});
    core::AssertTensorDtypes(min_bound, {core::Float32, core::Float64});

    const core::Tensor tmp = min_bound_.Clone();
    min_bound_ = min_bound.To(GetDtype());

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
    core::AssertTensorDevice(max_bound, GetDevice());
    core::AssertTensorShape(max_bound, {3});
    core::AssertTensorDtypes(max_bound, {core::Float32, core::Float64});

    const core::Tensor tmp = max_bound_.Clone();
    max_bound_ = max_bound.To(GetDtype());

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

    if (color.Max({0}).To(core::Float64).Item<double>() > 1.0 ||
        color.Min({0}).To(core::Float64).Item<double>() < 0.0) {
        utility::LogError(
                "The color must be in the range [0, 1], but for range [{}, "
                "{}].",
                color.Min({0}).To(core::Float64).Item<double>(),
                color.Max({0}).To(core::Float64).Item<double>());
    }

    color_ = color.To(GetDtype());
}

AxisAlignedBoundingBox &AxisAlignedBoundingBox::Translate(
        const core::Tensor &translation, bool relative) {
    core::AssertTensorDevice(translation, GetDevice());
    core::AssertTensorShape(translation, {3});
    core::AssertTensorDtypes(translation, {core::Float32, core::Float64});

    const core::Tensor translation_d = translation.To(GetDtype());
    if (relative) {
        min_bound_ += translation_d;
        max_bound_ += translation_d;
    } else {
        const core::Tensor half_extent = GetHalfExtent();
        min_bound_ = translation_d - half_extent;
        max_bound_ = translation_d + half_extent;
    }
    return *this;
}

AxisAlignedBoundingBox &AxisAlignedBoundingBox::Scale(
        double scale, const utility::optional<core::Tensor> &center) {
    core::Tensor center_d;
    if (!center.has_value()) {
        center_d = GetCenter();
    } else {
        center_d = center.value();
        core::AssertTensorDevice(center_d, GetDevice());
        core::AssertTensorShape(center_d, {3});
        core::AssertTensorDtypes(center_d, {core::Float32, core::Float64});
        center_d = center_d.To(GetDtype());
    }
    min_bound_ = center_d + scale * (min_bound_ - center_d);
    max_bound_ = center_d + scale * (max_bound_ - center_d);

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
    core::AssertTensorDtypes(points, {core::Float32, core::Float64});

    core::Tensor mask =
            core::Tensor::Zeros({points.GetLength()}, core::Bool, GetDevice());
    // Convert min_bound and max_bound to the same dtype as points.
    kernel::pointcloud::GetPointMaskWithinAABB(
            points, min_bound_.To(points.GetDtype()),
            max_bound_.To(points.GetDtype()), mask);

    return mask.NonZero().Flatten();
}

std::string AxisAlignedBoundingBox::ToString() const {
    return fmt::format("AxisAlignedBoundingBox[{}, {}]", GetDtype().ToString(),
                       GetDevice().ToString());
}

AxisAlignedBoundingBox AxisAlignedBoundingBox::CreateFromPoints(
        const core::Tensor &points) {
    core::AssertTensorShape(points, {utility::nullopt, 3});
    core::AssertTensorDtypes(points, {core::Float32, core::Float64});
    if (points.GetLength() <= 3) {
        utility::LogWarning("The points number is less than 3.");
        return AxisAlignedBoundingBox(points.GetDevice());
    } else {
        const core::Tensor min_bound = points.Min({0});
        const core::Tensor max_bound = points.Max({0});
        return AxisAlignedBoundingBox(min_bound, max_bound);
    }
}

open3d::geometry::AxisAlignedBoundingBox AxisAlignedBoundingBox::ToLegacy()
        const {
    open3d::geometry::AxisAlignedBoundingBox legacy_box;

    legacy_box.min_bound_ = core::eigen_converter::TensorToEigenVector3dVector(
            GetMinBound().Reshape({1, 3}))[0];
    legacy_box.max_bound_ = core::eigen_converter::TensorToEigenVector3dVector(
            GetMaxBound().Reshape({1, 3}))[0];
    legacy_box.color_ = core::eigen_converter::TensorToEigenVector3dVector(
            GetColor().Reshape({1, 3}))[0];
    return legacy_box;
}

OrientedBoundingBox AxisAlignedBoundingBox::GetOrientedBoundingBox() const {
    return OrientedBoundingBox::CreateFromAxisAlignedBoundingBox(*this);
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

OrientedBoundingBox::OrientedBoundingBox(const core::Device &device)
    : Geometry(Geometry::GeometryType::OrientedBoundingBox, 3),
      device_(device),
      dtype_(core::Float32),
      center_(core::Tensor::Zeros({3}, dtype_, device)),
      rotation_(core::Tensor::Eye(3, dtype_, device)),
      extent_(core::Tensor::Zeros({3}, dtype_, device)),
      color_(core::Tensor::Ones({3}, dtype_, device)) {}

OrientedBoundingBox::OrientedBoundingBox(const core::Tensor &center,
                                         const core::Tensor &rotation,
                                         const core::Tensor &extent)
    : OrientedBoundingBox([&]() {
          core::AssertTensorDevice(center, extent.GetDevice());
          core::AssertTensorDevice(rotation, extent.GetDevice());
          core::AssertTensorDtype(center, extent.GetDtype());
          core::AssertTensorDtype(rotation, extent.GetDtype());
          core::AssertTensorDtypes(extent, {core::Float32, core::Float64});
          core::AssertTensorShape(center, {3});
          core::AssertTensorShape(extent, {3});
          core::AssertTensorShape(rotation, {3, 3});
          return center.GetDevice();
      }()) {
    device_ = center.GetDevice();
    dtype_ = center.GetDtype();

    center_ = center;
    extent_ = extent;
    rotation_ = rotation;
    color_ = core::Tensor::Ones({3}, dtype_, device_);

    // Check if the bounding box is valid by checking the volume and the
    // orthogonality of rotation.
    if (Volume() < 0 ||
        !rotation_.T().AllClose(rotation.Inverse(), 1e-5, 1e-5)) {
        utility::LogError(
                "Invalid oriented bounding box. Please make sure the values of "
                "extent are all positive and the rotation matrix is "
                "othogonal.");
    }
}

OrientedBoundingBox OrientedBoundingBox::To(const core::Device &device,
                                            bool copy) const {
    if (!copy && GetDevice() == device) {
        return *this;
    }
    OrientedBoundingBox box(device);
    box.SetCenter(center_.To(device, true));
    box.SetRotation(rotation_.To(device, true));
    box.SetExtent(extent_.To(device, true));
    box.SetColor(color_.To(device, true));
    return box;
}

OrientedBoundingBox &OrientedBoundingBox::Clear() {
    center_ = core::Tensor::Zeros({3}, GetDtype(), GetDevice());
    extent_ = core::Tensor::Zeros({3}, GetDtype(), GetDevice());
    rotation_ = core::Tensor::Eye(3, GetDtype(), GetDevice());
    color_ = core::Tensor::Ones({3}, GetDtype(), GetDevice());
    return *this;
}

void OrientedBoundingBox::SetCenter(const core::Tensor &center) {
    core::AssertTensorDevice(center, GetDevice());
    core::AssertTensorShape(center, {3});
    core::AssertTensorDtypes(center, {core::Float32, core::Float64});

    center_ = center.To(GetDtype());
}

void OrientedBoundingBox::SetExtent(const core::Tensor &extent) {
    core::AssertTensorDevice(extent, GetDevice());
    core::AssertTensorShape(extent, {3});
    core::AssertTensorDtypes(extent, {core::Float32, core::Float64});

    if (extent.Min({0}).To(core::Float64).Item<double>() <= 0) {
        utility::LogError(
                "Invalid oriented bounding box. Please make sure the values of "
                "extent are all positive.");
    }

    extent_ = extent.To(GetDtype());
}

void OrientedBoundingBox::SetRotation(const core::Tensor &rotation) {
    core::AssertTensorDevice(rotation, GetDevice());
    core::AssertTensorShape(rotation, {3, 3});
    core::AssertTensorDtypes(rotation, {core::Float32, core::Float64});

    if (!rotation.T().AllClose(rotation.Inverse(), 1e-5, 1e-5)) {
        utility::LogWarning(
                "Invalid oriented bounding box. Please make sure the rotation "
                "matrix is orthogonal.");
    } else {
        rotation_ = rotation.To(GetDtype());
    }
}

void OrientedBoundingBox::SetColor(const core::Tensor &color) {
    core::AssertTensorDevice(color, GetDevice());
    core::AssertTensorShape(color, {3});
    if (color.Max({0}).To(core::Float64).Item<double>() > 1.0 ||
        color.Min({0}).To(core::Float64).Item<double>() < 0.0) {
        utility::LogError(
                "The color must be in the range [0, 1], but for range [{}, "
                "{}].",
                color.Min({0}).To(core::Float64).Item<double>(),
                color.Max({0}).To(core::Float64).Item<double>());
    }

    color_ = color.To(GetDtype());
}

core::Tensor OrientedBoundingBox::GetMinBound() const {
    return GetBoxPoints().Min({0});
}

core::Tensor OrientedBoundingBox::GetMaxBound() const {
    return GetBoxPoints().Max({0});
}

core::Tensor OrientedBoundingBox::GetBoxPoints() const {
    const t::geometry::AxisAlignedBoundingBox aabb(GetExtent() * -0.5,
                                                   GetExtent() * 0.5);
    return aabb.GetBoxPoints().Matmul(GetRotation().T()).Add(GetCenter());
}

OrientedBoundingBox &OrientedBoundingBox::Translate(
        const core::Tensor &translation, bool relative) {
    core::AssertTensorDevice(translation, GetDevice());
    core::AssertTensorShape(translation, {3});
    core::AssertTensorDtypes(translation, {core::Float32, core::Float64});

    const core::Tensor translation_d = translation.To(GetDtype());
    if (relative) {
        center_ += translation_d;
    } else {
        center_ = translation_d;
    }
    return *this;
}

OrientedBoundingBox &OrientedBoundingBox::Rotate(
        const core::Tensor &rotation,
        const utility::optional<core::Tensor> &center) {
    core::AssertTensorDevice(rotation, GetDevice());
    core::AssertTensorShape(rotation, {3, 3});
    core::AssertTensorDtypes(rotation, {core::Float32, core::Float64});

    if (!rotation.T().AllClose(rotation.Inverse(), 1e-5, 1e-5)) {
        utility::LogWarning(
                "Invalid rotation matrix. Please make sure the rotation "
                "matrix is orthogonal.");
        return *this;
    }

    const core::Tensor rotation_d = rotation.To(GetDtype());
    rotation_ = rotation_d.Matmul(rotation_);
    if (center.has_value()) {
        core::AssertTensorDevice(center.value(), GetDevice());
        core::AssertTensorShape(center.value(), {3});
        core::AssertTensorDtypes(center.value(),
                                 {core::Float32, core::Float64});

        core::Tensor center_d = center.value().To(GetDtype());
        center_ = rotation_d.Matmul(center_ - center_d).Flatten() + center_d;
    }

    return *this;
}

OrientedBoundingBox &OrientedBoundingBox::Transform(
        const core::Tensor &transformation) {
    core::AssertTensorDevice(transformation, GetDevice());
    core::AssertTensorShape(transformation, {4, 4});
    core::AssertTensorDtypes(transformation, {core::Float32, core::Float64});

    const core::Tensor transformation_d = transformation.To(GetDtype());
    Rotate(transformation_d.GetItem({core::TensorKey::Slice(0, 3, 1),
                                     core::TensorKey::Slice(0, 3, 1)}));
    Translate(transformation_d
                      .GetItem({core::TensorKey::Slice(0, 3, 1),
                                core::TensorKey::Index(3)})
                      .Flatten());
    return *this;
}

OrientedBoundingBox &OrientedBoundingBox::Scale(
        const double scale, const utility::optional<core::Tensor> &center) {
    extent_ *= scale;
    if (center.has_value()) {
        core::Tensor center_d = center.value();
        core::AssertTensorDevice(center_d, GetDevice());
        core::AssertTensorShape(center_d, {3});
        core::AssertTensorDtypes(center_d, {core::Float32, core::Float64});

        center_d = center_d.To(GetDtype());
        center_ = scale * (center_ - center_d) + center_d;
    }
    return *this;
}

core::Tensor OrientedBoundingBox::GetPointIndicesWithinBoundingBox(
        const core::Tensor &points) const {
    core::AssertTensorDevice(points, GetDevice());
    core::AssertTensorShape(points, {utility::nullopt, 3});
    core::AssertTensorDtypes(points, {core::Float32, core::Float64});

    core::Tensor mask =
            core::Tensor::Zeros({points.GetLength()}, core::Bool, GetDevice());
    // Convert center, rotation and same to the same dtype as points.
    kernel::pointcloud::GetPointMaskWithinOBB(
            points, center_.To(points.GetDtype()),
            rotation_.To(points.GetDtype()), extent_.To(points.GetDtype()),
            mask);

    return mask.NonZero().Flatten();
}

std::string OrientedBoundingBox::ToString() const {
    return fmt::format("OrientedBoundingBox[{}, {}]", GetDtype().ToString(),
                       GetDevice().ToString());
}

open3d::geometry::OrientedBoundingBox OrientedBoundingBox::ToLegacy() const {
    open3d::geometry::OrientedBoundingBox legacy_box;

    legacy_box.center_ = core::eigen_converter::TensorToEigenVector3dVector(
            GetCenter().Reshape({1, 3}))[0];
    legacy_box.extent_ = core::eigen_converter::TensorToEigenVector3dVector(
            GetExtent().Reshape({1, 3}))[0];
    legacy_box.R_ = core::eigen_converter::TensorToEigenMatrixXd(GetRotation());
    legacy_box.color_ = core::eigen_converter::TensorToEigenVector3dVector(
            GetColor().Reshape({1, 3}))[0];
    return legacy_box;
}

AxisAlignedBoundingBox OrientedBoundingBox::GetAxisAlignedBoundingBox() const {
    return AxisAlignedBoundingBox::CreateFromPoints(GetBoxPoints());
}

OrientedBoundingBox OrientedBoundingBox::CreateFromAxisAlignedBoundingBox(
        const AxisAlignedBoundingBox &aabb) {
    OrientedBoundingBox box(
            aabb.GetCenter(),
            core::Tensor::Eye(3, aabb.GetDtype(), aabb.GetDevice()),
            aabb.GetExtent());
    return box;
}

OrientedBoundingBox OrientedBoundingBox::FromLegacy(
        const open3d::geometry::OrientedBoundingBox &box,
        const core::Dtype &dtype,
        const core::Device &device) {
    if (dtype != core::Float32 && dtype != core::Float64) {
        utility::LogError(
                "Got data-type {}, but the supported data-type of the bounding "
                "box are Float32 and Float64.",
                dtype.ToString());
    }

    OrientedBoundingBox t_box(
            core::eigen_converter::EigenMatrixToTensor(box.center_)
                    .Flatten()
                    .To(device, dtype),
            core::eigen_converter::EigenMatrixToTensor(box.R_).To(device,
                                                                  dtype),
            core::eigen_converter::EigenMatrixToTensor(box.extent_)
                    .Flatten()
                    .To(device, dtype));

    t_box.SetColor(core::eigen_converter::EigenMatrixToTensor(box.color_)
                           .Flatten()
                           .To(device, dtype));
    return t_box;
}

OrientedBoundingBox OrientedBoundingBox::CreateFromPoints(
        const core::Tensor &points, bool robust) {
    core::AssertTensorShape(points, {utility::nullopt, 3});
    core::AssertTensorDtypes(points, {core::Float32, core::Float64});
    return OrientedBoundingBox::FromLegacy(
            open3d::geometry::OrientedBoundingBox::CreateFromPoints(
                    core::eigen_converter::TensorToEigenVector3dVector(points),
                    robust),
            points.GetDtype(), points.GetDevice());
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
