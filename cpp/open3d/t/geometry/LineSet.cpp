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

#include "open3d/t/geometry/LineSet.h"

#include <string>

#include "open3d/core/EigenConverter.h"
#include "open3d/core/ShapeUtil.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/TensorCheck.h"
#include "open3d/core/linalg/Matmul.h"
#include "open3d/t/geometry/TensorMap.h"
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

}  // namespace geometry
}  // namespace t
}  // namespace open3d
