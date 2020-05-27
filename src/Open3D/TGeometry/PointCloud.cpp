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

#include "Open3D/TGeometry/PointCloud.h"

#include <Eigen/Core>
#include <unordered_map>

#include "Open3D/Core/ShapeUtil.h"
#include "Open3D/Core/Tensor.h"
#include "Open3D/Core/TensorList.h"

namespace open3d {
namespace tgeometry {

PointCloud &PointCloud::Clear() {
    point_dict_.clear();
    return *this;
}

bool PointCloud::IsEmpty() const { return !HasPoints(); }

Tensor PointCloud::GetMinBound() const {
    point_dict_.at("points").AssertShape({3});
    return point_dict_.at("points").AsTensor().Min({0});
}

Tensor PointCloud::GetMaxBound() const {
    point_dict_.at("points").AssertShape({3});
    return point_dict_.at("points").AsTensor().Max({0});
}

Tensor PointCloud::GetCenter() const {
    point_dict_.at("points").AssertShape({3});
    return point_dict_.at("points").AsTensor().Mean({0});
}

PointCloud &PointCloud::Transform(const Tensor &transformation) {
    utility::LogError("Unimplemented");
    return *this;
}

PointCloud &PointCloud::Translate(const Tensor &translation, bool relative) {
    shape_util::AssertShape(translation, {3},
                            "translation must have shape (3,)");
    Tensor transform = translation.Copy();
    if (!relative) {
        transform -= GetCenter();
    }
    point_dict_.at("points").AsTensor() += transform;
    return *this;
}

PointCloud &PointCloud::Scale(double scale, const Tensor &center) {
    shape_util::AssertShape(center, {3}, "center must have shape (3,)");
    point_dict_.at("points") = TensorList(
            (point_dict_.at("points").AsTensor() - center) * scale + center);
    return *this;
}

PointCloud &PointCloud::Rotate(const Tensor &R, const Tensor &center) {
    utility::LogError("Unimplemented");
    return *this;
}

}  // namespace tgeometry
}  // namespace open3d
