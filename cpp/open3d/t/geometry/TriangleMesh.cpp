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

#include "open3d/t/geometry/TriangleMesh.h"

#include <Eigen/Core>
#include <string>
#include <unordered_map>

#include "open3d/core/EigenConverter.h"
#include "open3d/core/ShapeUtil.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/kernel/PointCloud.h"
#include "open3d/t/geometry/kernel/Transform.h"

namespace open3d {
namespace t {
namespace geometry {

TriangleMesh::TriangleMesh(const core::Device &device)
    : Geometry(Geometry::GeometryType::TriangleMesh, 3),
      device_(device),
      vertex_attr_(TensorMap("vertices")),
      triangle_attr_(TensorMap("triangles")) {}

TriangleMesh::TriangleMesh(const core::Tensor &vertices,
                           const core::Tensor &triangles)
    : TriangleMesh([&]() {
          if (vertices.GetDevice() != triangles.GetDevice()) {
              utility::LogError(
                      "vertices' device {} does not match triangles' device "
                      "{}.",
                      vertices.GetDevice().ToString(),
                      triangles.GetDevice().ToString());
          }
          return vertices.GetDevice();
      }()) {
    SetVertices(vertices);
    SetTriangles(triangles);
}

TriangleMesh &TriangleMesh::Transform(const core::Tensor &transformation) {
    kernel::transform::TransformPoints(transformation, GetVertices());
    if (HasVertexNormals()) {
        kernel::transform::TransformNormals(transformation, GetVertexNormals());
    }
    if (HasTriangleNormals()) {
        kernel::transform::TransformNormals(transformation,
                                            GetTriangleNormals());
    }

    return *this;
}

TriangleMesh &TriangleMesh::Translate(const core::Tensor &translation,
                                      bool relative) {
    translation.AssertShape({3});
    translation.AssertDevice(device_);

    core::Tensor transform = translation;
    if (!relative) {
        transform -= GetCenter();
    }
    GetVertices() += transform;
    return *this;
}

TriangleMesh &TriangleMesh::Scale(double scale, const core::Tensor &center) {
    center.AssertShape({3});
    center.AssertDevice(device_);

    core::Tensor points = GetVertices();
    points.Sub_(center).Mul_(scale).Add_(center);
    return *this;
}

TriangleMesh &TriangleMesh::Rotate(const core::Tensor &R,
                                   const core::Tensor &center) {
    kernel::transform::RotatePoints(R, GetVertices(), center);
    if (HasVertexNormals()) {
        kernel::transform::RotateNormals(R, GetVertexNormals());
    }
    if (HasTriangleNormals()) {
        kernel::transform::RotateNormals(R, GetTriangleNormals());
    }
    return *this;
}

geometry::TriangleMesh TriangleMesh::FromLegacy(
        const open3d::geometry::TriangleMesh &mesh_legacy,
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

    TriangleMesh mesh(device);
    if (mesh_legacy.HasVertices()) {
        mesh.SetVertices(core::eigen_converter::EigenVector3dVectorToTensor(
                mesh_legacy.vertices_, float_dtype, device));
    } else {
        utility::LogWarning("Creating from empty legacy TriangleMesh.");
    }
    if (mesh_legacy.HasVertexColors()) {
        mesh.SetVertexColors(core::eigen_converter::EigenVector3dVectorToTensor(
                mesh_legacy.vertex_colors_, float_dtype, device));
    }
    if (mesh_legacy.HasVertexNormals()) {
        mesh.SetVertexNormals(
                core::eigen_converter::EigenVector3dVectorToTensor(
                        mesh_legacy.vertex_normals_, float_dtype, device));
    }
    if (mesh_legacy.HasTriangles()) {
        mesh.SetTriangles(core::eigen_converter::EigenVector3iVectorToTensor(
                mesh_legacy.triangles_, int_dtype, device));
    }
    if (mesh_legacy.HasTriangleNormals()) {
        mesh.SetTriangleNormals(
                core::eigen_converter::EigenVector3dVectorToTensor(
                        mesh_legacy.triangle_normals_, float_dtype, device));
    }
    return mesh;
}

open3d::geometry::TriangleMesh TriangleMesh::ToLegacy() const {
    open3d::geometry::TriangleMesh mesh_legacy;
    if (HasVertices()) {
        mesh_legacy.vertices_ =
                core::eigen_converter::TensorToEigenVector3dVector(
                        GetVertices());
    }
    if (HasVertexColors()) {
        mesh_legacy.vertex_colors_ =
                core::eigen_converter::TensorToEigenVector3dVector(
                        GetVertexColors());
    }
    if (HasVertexNormals()) {
        mesh_legacy.vertex_normals_ =
                core::eigen_converter::TensorToEigenVector3dVector(
                        GetVertexNormals());
    }
    if (HasTriangles()) {
        mesh_legacy.triangles_ =
                core::eigen_converter::TensorToEigenVector3iVector(
                        GetTriangles());
    }
    if (HasTriangleNormals()) {
        mesh_legacy.triangle_normals_ =
                core::eigen_converter::TensorToEigenVector3dVector(
                        GetTriangleNormals());
    }

    return mesh_legacy;
}

TriangleMesh TriangleMesh::To(const core::Device &device, bool copy) const {
    if (!copy && GetDevice() == device) {
        return *this;
    }
    TriangleMesh mesh(device);
    for (const auto &kv : triangle_attr_) {
        mesh.SetTriangleAttr(kv.first, kv.second.To(device, /*copy=*/true));
    }
    for (const auto &kv : vertex_attr_) {
        mesh.SetVertexAttr(kv.first, kv.second.To(device, /*copy=*/true));
    }
    return mesh;
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
