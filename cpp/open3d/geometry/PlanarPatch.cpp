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

#include "open3d/geometry/PlanarPatch.h"

#include <algorithm>
#include <numeric>
#include <vector>

#include <Eigen/Dense>

#include "open3d/geometry/BoundingVolume.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/utility/Console.h"

namespace open3d {
namespace geometry {

PlanarPatch& PlanarPatch::Clear() { return *this; }

bool PlanarPatch::IsEmpty() const { return false; }

Eigen::Vector3d PlanarPatch::GetMinBound() const { return Eigen::Vector3d::Zero(); }

Eigen::Vector3d PlanarPatch::GetMaxBound() const { return Eigen::Vector3d::Zero(); }

Eigen::Vector3d PlanarPatch::GetCenter() const { return Eigen::Vector3d::Zero(); }

AxisAlignedBoundingBox PlanarPatch::GetAxisAlignedBoundingBox() const {
    AxisAlignedBoundingBox box;
    box.min_bound_ = GetMinBound();
    box.max_bound_ = GetMaxBound();
    return box;
}

OrientedBoundingBox PlanarPatch::GetOrientedBoundingBox() const {
    return OrientedBoundingBox::CreateFromAxisAlignedBoundingBox(
            GetAxisAlignedBoundingBox());
}

PlanarPatch& PlanarPatch::Transform(const Eigen::Matrix4d& transformation) {
    utility::LogError("Not implemented");
    return *this;
}

PlanarPatch& PlanarPatch::Translate(const Eigen::Vector3d& translation,
                                  bool relative) {
    utility::LogError("Not implemented");
    return *this;
}

PlanarPatch& PlanarPatch::Scale(const double scale,
                              const Eigen::Vector3d& center) {
    utility::LogError("Not implemented");
    return *this;
}

PlanarPatch& PlanarPatch::Rotate(const Eigen::Matrix3d& R,
                               const Eigen::Vector3d& center) {
    utility::LogError("Not implemented");
    return *this;
}

}  // namespace geometry
}  // namespace open3d
