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

#include "open3d/geometry/MeshBase.h"

#include <Eigen/Dense>
#include <numeric>
#include <queue>
#include <tuple>

#include "open3d/geometry/BoundingVolume.h"
#include "open3d/geometry/IntersectionTest.h"
#include "open3d/geometry/KDTreeFlann.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/Qhull.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace geometry {

MeshBase &MeshBase::Clear() {
    vertices_.clear();
    vertex_normals_.clear();
    vertex_colors_.clear();
    return *this;
}

bool MeshBase::IsEmpty() const { return !HasVertices(); }

Eigen::Vector3d MeshBase::GetMinBound() const {
    return ComputeMinBound(vertices_);
}

Eigen::Vector3d MeshBase::GetMaxBound() const {
    return ComputeMaxBound(vertices_);
}

Eigen::Vector3d MeshBase::GetCenter() const { return ComputeCenter(vertices_); }

AxisAlignedBoundingBox MeshBase::GetAxisAlignedBoundingBox() const {
    return AxisAlignedBoundingBox::CreateFromPoints(vertices_);
}

OrientedBoundingBox MeshBase::GetOrientedBoundingBox(bool robust) const {
    return OrientedBoundingBox::CreateFromPoints(vertices_, robust);
}

MeshBase &MeshBase::Transform(const Eigen::Matrix4d &transformation) {
    TransformPoints(transformation, vertices_);
    TransformNormals(transformation, vertex_normals_);
    return *this;
}

MeshBase &MeshBase::Translate(const Eigen::Vector3d &translation,
                              bool relative) {
    TranslatePoints(translation, vertices_, relative);
    return *this;
}

MeshBase &MeshBase::Scale(const double scale, const Eigen::Vector3d &center) {
    ScalePoints(scale, vertices_, center);
    return *this;
}

MeshBase &MeshBase::Rotate(const Eigen::Matrix3d &R,
                           const Eigen::Vector3d &center) {
    RotatePoints(R, vertices_, center);
    RotateNormals(R, vertex_normals_);
    return *this;
}

MeshBase &MeshBase::operator+=(const MeshBase &mesh) {
    if (mesh.IsEmpty()) return (*this);
    size_t old_vert_num = vertices_.size();
    size_t add_vert_num = mesh.vertices_.size();
    size_t new_vert_num = old_vert_num + add_vert_num;
    if ((!HasVertices() || HasVertexNormals()) && mesh.HasVertexNormals()) {
        vertex_normals_.resize(new_vert_num);
        for (size_t i = 0; i < add_vert_num; i++)
            vertex_normals_[old_vert_num + i] = mesh.vertex_normals_[i];
    } else {
        vertex_normals_.clear();
    }
    if ((!HasVertices() || HasVertexColors()) && mesh.HasVertexColors()) {
        vertex_colors_.resize(new_vert_num);
        for (size_t i = 0; i < add_vert_num; i++)
            vertex_colors_[old_vert_num + i] = mesh.vertex_colors_[i];
    } else {
        vertex_colors_.clear();
    }
    vertices_.resize(new_vert_num);
    for (size_t i = 0; i < add_vert_num; i++)
        vertices_[old_vert_num + i] = mesh.vertices_[i];
    return (*this);
}

MeshBase MeshBase::operator+(const MeshBase &mesh) const {
    return (MeshBase(*this) += mesh);
}

std::tuple<std::shared_ptr<TriangleMesh>, std::vector<size_t>>
MeshBase::ComputeConvexHull() const {
    return Qhull::ComputeConvexHull(vertices_);
}

}  // namespace geometry
}  // namespace open3d
