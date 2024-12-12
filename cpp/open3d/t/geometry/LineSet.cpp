// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/geometry/LineSet.h"

#include <string>

#include "open3d/core/Dtype.h"
#include "open3d/core/EigenConverter.h"
#include "open3d/core/ShapeUtil.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/TensorCheck.h"
#include "open3d/core/linalg/Matmul.h"
#include "open3d/t/geometry/TensorMap.h"
#include "open3d/t/geometry/VtkUtils.h"
#include "open3d/t/geometry/kernel/Transform.h"

namespace open3d {
namespace t {
namespace geometry {

LineSet::LineSet(const core::Device &device)
    : Geometry(Geometry::GeometryType::LineSet, 3),
      device_(device),
      point_attr_(TensorMap("positions")),
      line_attr_(TensorMap("indices")) {}

LineSet::LineSet(const core::Tensor &point_positions,
                 const core::Tensor &line_indices)
    : LineSet([&]() {
          core::AssertTensorDevice(line_indices, point_positions.GetDevice());
          return point_positions.GetDevice();
      }()) {
    SetPointPositions(point_positions);
    SetLineIndices(line_indices);
}

LineSet LineSet::To(const core::Device &device, bool copy) const {
    if (!copy && GetDevice() == device) {
        return *this;
    }
    LineSet lineset(device);
    for (const auto &kv : line_attr_) {
        lineset.SetLineAttr(kv.first, kv.second.To(device, /*copy=*/true));
    }
    for (const auto &kv : point_attr_) {
        lineset.SetPointAttr(kv.first, kv.second.To(device, /*copy=*/true));
    }
    return lineset;
}

std::string LineSet::ToString() const {
    std::string str = fmt::format("LineSet on {}\n", GetDevice().ToString());
    if (point_attr_.size() == 0) {
        str += "[0 points ()] Attributes: None.";
    } else {
        str += fmt::format(
                "[{} points ({})] Attributes:", GetPointPositions().GetShape(0),
                GetPointPositions().GetDtype().ToString());
    }
    if (point_attr_.size() == 1) {
        str += " None.";
    } else {
        for (const auto &keyval : point_attr_) {
            if (keyval.first == "positions") continue;
            str += fmt::format(" {} (dtype = {}, shape = {}),", keyval.first,
                               keyval.second.GetDtype().ToString(),
                               keyval.second.GetShape().ToString());
        }
        str[str.size() - 1] = '.';
    }
    if (line_attr_.size() == 0) {
        str += "\n[0 lines ()] Attributes: None.";
    } else {
        str += fmt::format(
                "\n[{} lines ({})] Attributes:", GetLineIndices().GetShape(0),
                GetLineIndices().GetDtype().ToString());
    }
    if (line_attr_.size() == 1) {
        str += " None.";
    } else {
        for (const auto &keyval : line_attr_) {
            if (keyval.first == "indices") continue;
            str += fmt::format(" {} (dtype = {}, shape = {}),", keyval.first,
                               keyval.second.GetDtype().ToString(),
                               keyval.second.GetShape().ToString());
        }
        str[str.size() - 1] = '.';
    }
    return str;
}

LineSet &LineSet::Transform(const core::Tensor &transformation) {
    core::AssertTensorShape(transformation, {4, 4});
    kernel::transform::TransformPoints(transformation, GetPointPositions());
    return *this;
}

LineSet &LineSet::Translate(const core::Tensor &translation, bool relative) {
    core::AssertTensorShape(translation, {3});

    core::Tensor transform =
            translation.To(GetDevice(), GetPointPositions().GetDtype());

    if (!relative) {
        transform -= GetCenter();
    }
    GetPointPositions() += transform;
    return *this;
}

LineSet &LineSet::Scale(double scale, const core::Tensor &center) {
    core::AssertTensorShape(center, {3});

    const core::Tensor center_d =
            center.To(GetDevice(), GetPointPositions().GetDtype());

    GetPointPositions().Sub_(center_d).Mul_(scale).Add_(center_d);
    return *this;
}

LineSet &LineSet::Rotate(const core::Tensor &R, const core::Tensor &center) {
    core::AssertTensorShape(R, {3, 3});
    core::AssertTensorShape(center, {3});
    kernel::transform::RotatePoints(R, GetPointPositions(), center);
    return *this;
}

geometry::LineSet LineSet::FromLegacy(
        const open3d::geometry::LineSet &lineset_legacy,
        core::Dtype float_dtype,
        core::Dtype int_dtype,
        const core::Device &device) {
    if (float_dtype != core::Float32 && float_dtype != core::Float64) {
        utility::LogError("float_dtype must be Float32 or Float64, but got {}.",
                          float_dtype.ToString());
    }
    if (int_dtype != core::Int32 && int_dtype != core::Int64) {
        utility::LogError("int_dtype must be Int32 or Int64, but got {}.",
                          int_dtype.ToString());
    }

    LineSet lineset(device);
    if (lineset_legacy.HasPoints()) {
        lineset.SetPointPositions(
                core::eigen_converter::EigenVector3dVectorToTensor(
                        lineset_legacy.points_, float_dtype, device));
    } else {
        utility::LogWarning("Creating from empty legacy LineSet.");
    }
    if (lineset_legacy.HasLines()) {
        lineset.SetLineIndices(
                core::eigen_converter::EigenVector2iVectorToTensor(
                        lineset_legacy.lines_, int_dtype, device));
    } else {
        utility::LogWarning("Creating from legacy LineSet with no lines.");
    }
    if (lineset_legacy.HasColors()) {
        lineset.SetLineColors(
                core::eigen_converter::EigenVector3dVectorToTensor(
                        lineset_legacy.colors_, float_dtype, device));
    }
    return lineset;
}

open3d::geometry::LineSet LineSet::ToLegacy() const {
    open3d::geometry::LineSet lineset_legacy;
    if (HasPointPositions()) {
        lineset_legacy.points_ =
                core::eigen_converter::TensorToEigenVector3dVector(
                        GetPointPositions());
    }
    if (HasLineIndices()) {
        lineset_legacy.lines_ =
                core::eigen_converter::TensorToEigenVector2iVector(
                        GetLineIndices());
    }
    if (HasLineColors()) {
        lineset_legacy.colors_ =
                core::eigen_converter::TensorToEigenVector3dVector(
                        GetLineColors());
    }
    return lineset_legacy;
}

AxisAlignedBoundingBox LineSet::GetAxisAlignedBoundingBox() const {
    return AxisAlignedBoundingBox::CreateFromPoints(GetPointPositions());
}

TriangleMesh LineSet::ExtrudeRotation(double angle,
                                      const core::Tensor &axis,
                                      int resolution,
                                      double translation,
                                      bool capping) const {
    using namespace vtkutils;
    return ExtrudeRotationTriangleMesh(*this, angle, axis, resolution,
                                       translation, capping);
}

TriangleMesh LineSet::ExtrudeLinear(const core::Tensor &vector,
                                    double scale,
                                    bool capping) const {
    using namespace vtkutils;
    return ExtrudeLinearTriangleMesh(*this, vector, scale, capping);
}

OrientedBoundingBox LineSet::GetOrientedBoundingBox() const {
    return OrientedBoundingBox::CreateFromPoints(GetPointPositions());
}

LineSet &LineSet::PaintUniformColor(const core::Tensor &color) {
    core::AssertTensorShape(color, {3});
    core::Tensor clipped_color = color.To(GetDevice());
    if (color.GetDtype() == core::Float32 ||
        color.GetDtype() == core::Float64) {
        clipped_color = clipped_color.Clip(0.0f, 1.0f);
    }
    core::Tensor ls_colors =
            core::Tensor::Empty({GetLineIndices().GetLength(), 3},
                                clipped_color.GetDtype(), GetDevice());
    ls_colors.AsRvalue() = clipped_color;
    SetLineColors(ls_colors);

    return *this;
}

LineSet LineSet::CreateCameraVisualization(int view_width_px,
                                           int view_height_px,
                                           const core::Tensor &intrinsic_in,
                                           const core::Tensor &extrinsic_in,
                                           double scale,
                                           const core::Tensor &color) {
    core::AssertTensorShape(intrinsic_in, {3, 3});
    core::AssertTensorShape(extrinsic_in, {4, 4});
    core::Tensor intrinsic = intrinsic_in.To(core::Float32, "CPU:0");
    core::Tensor extrinsic = extrinsic_in.To(core::Float32, "CPU:0");

    // Calculate points for camera visualization
    float w(view_width_px), h(view_height_px), s(scale);
    float fx = intrinsic[0][0].Item<float>(),
          fy = intrinsic[1][1].Item<float>(),
          cx = intrinsic[0][2].Item<float>(),
          cy = intrinsic[1][2].Item<float>();
    core::Tensor points = core::Tensor::Init<float>({{0.f, 0.f, 0.f},  // origin
                                                     {0.f, 0.f, s},
                                                     {w * s, 0.f, s},
                                                     {w * s, h * s, s},
                                                     {0.f, h * s, s},
                                                     // Add XYZ axes
                                                     {fx * s, 0.f, 0.f},
                                                     {0.f, fy * s, 0.f},
                                                     {cx * s, cy * s, s}});
    points = (intrinsic.Inverse().Matmul(points.T()) -
              extrinsic.Slice(0, 0, 3).Slice(1, 3, 4))
                     .T()
                     .Matmul(extrinsic.Slice(0, 0, 3).Slice(1, 0, 3));

    // Add lines for camera frame and XYZ axes
    core::Tensor lines = core::Tensor::Init<int>({{0, 1},
                                                  {0, 2},
                                                  {0, 3},
                                                  {0, 4},
                                                  {1, 2},
                                                  {2, 3},
                                                  {3, 4},
                                                  {4, 1},
                                                  // Add XYZ axes
                                                  {0, 5},
                                                  {0, 6},
                                                  {0, 7}});

    LineSet lineset(points, lines);
    if (color.NumElements() == 3) {
        lineset.PaintUniformColor(color);
    } else {
        lineset.PaintUniformColor(core::Tensor::Init<float>({0.f, 0.f, 1.f}));
    }
    auto &lscolors = lineset.GetLineColors();
    lscolors[8] = core::Tensor::Init<float>({1.f, 0.f, 0.f});   // Red
    lscolors[9] = core::Tensor::Init<float>({0.f, 1.f, 0.f});   // Green
    lscolors[10] = core::Tensor::Init<float>({0.f, 0.f, 1.f});  // Blue

    return lineset;
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
