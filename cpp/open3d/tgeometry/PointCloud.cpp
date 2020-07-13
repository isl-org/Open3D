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

#include "open3d/tgeometry/PointCloud.h"

#include <Eigen/Core>
#include <unordered_map>

#include "open3d/core/ShapeUtil.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/TensorList.h"

namespace open3d {
namespace tgeometry {

PointCloud::PointCloud(const core::Tensor &points_tensor)
    : Geometry3D(Geometry::GeometryType::PointCloud),
      dtype_(points_tensor.GetDtype()),
      device_(points_tensor.GetDevice()) {
    auto shape = points_tensor.GetShape();
    if (shape[1] != 3) {
        utility::LogError("PointCloud must be constructed from (N, 3) points.");
    }
    point_dict_.emplace("points", core::TensorList::FromTensor(points_tensor));
}

PointCloud::PointCloud(
        const std::unordered_map<std::string, core::TensorList> &point_dict)
    : Geometry3D(Geometry::GeometryType::PointCloud) {
    auto it = point_dict.find("points");
    if (it == point_dict.end()) {
        utility::LogError("PointCloud must include key \"points\".");
    }

    dtype_ = it->second.GetDtype();
    device_ = it->second.GetDevice();

    auto shape = it->second.GetElementShape();
    if (shape[0] != 3) {
        utility::LogError("PointCloud must be constructed from (N, 3) points.");
    }

    for (auto kv : point_dict) {
        if (device_ != kv.second.GetDevice()) {
            utility::LogError("TensorList device mismatch!");
        }
        point_dict_.emplace(kv.first, kv.second);
    }
}

core::TensorList &PointCloud::operator[](const std::string &key) {
    auto it = point_dict_.find(key);
    if (it == point_dict_.end()) {
        utility::LogError("Unknown key {} in PointCloud point_dict_.", key);
    }

    return it->second;
}

void PointCloud::SyncPushBack(
        const std::unordered_map<std::string, core::Tensor> &point_struct) {
    // Check if "point"" exists
    auto it = point_struct.find("points");
    if (it == point_struct.end()) {
        utility::LogError("Point must include key \"points\".");
    }

    auto size = point_dict_.find("points")->second.GetSize();
    for (auto kv : point_struct) {
        // Check existance of key in point_dict
        auto it = point_dict_.find(kv.first);
        if (it == point_dict_.end()) {
            utility::LogError("Unknown key {} in PointCloud dictionary.",
                              kv.first);
        }

        // Check size consistency
        auto size_it = it->second.GetSize();
        if (size_it != size) {
            utility::LogError("Size mismatch ({}, {}) between ({}, {}).",
                              "points", size, kv.first, size_it);
        }
        it->second.PushBack(kv.second);
    }
}

PointCloud &PointCloud::Clear() {
    point_dict_.clear();
    return *this;
}

bool PointCloud::IsEmpty() const { return !HasPoints(); }

core::Tensor PointCloud::GetMinBound() const {
    point_dict_.at("points").AssertElementShape({3});
    return point_dict_.at("points").AsTensor().Min({0});
}

core::Tensor PointCloud::GetMaxBound() const {
    point_dict_.at("points").AssertElementShape({3});
    return point_dict_.at("points").AsTensor().Max({0});
}

core::Tensor PointCloud::GetCenter() const {
    point_dict_.at("points").AssertElementShape({3});
    return point_dict_.at("points").AsTensor().Mean({0});
}

PointCloud &PointCloud::Transform(const core::Tensor &transformation) {
    utility::LogError("Unimplemented");
    return *this;
}

PointCloud &PointCloud::Translate(const core::Tensor &translation,
                                  bool relative) {
    core::shape_util::AssertShape(translation, {3},
                                  "translation must have shape (3,)");
    core::Tensor transform = translation.Copy();
    if (!relative) {
        transform -= GetCenter();
    }
    point_dict_.at("points").AsTensor() += transform;
    return *this;
}

PointCloud &PointCloud::Scale(double scale, const core::Tensor &center) {
    core::shape_util::AssertShape(center, {3}, "center must have shape (3,)");
    point_dict_.at("points").AsTensor() =
            (point_dict_.at("points").AsTensor() - center) * scale + center;
    return *this;
}

PointCloud &PointCloud::Rotate(const core::Tensor &R,
                               const core::Tensor &center) {
    utility::LogError("Unimplemented");
    return *this;
}

}  // namespace tgeometry
}  // namespace open3d
