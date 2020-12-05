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

#include "open3d/t/geometry/TriangleMesh.h"

#include <Eigen/Core>
#include <string>
#include <unordered_map>

#include "open3d/core/EigenConverter.h"
#include "open3d/core/ShapeUtil.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/TensorList.h"

namespace open3d {
namespace t {
namespace geometry {

TriangleMesh::TriangleMesh(core::Dtype vertex_dtype,
                           core::Dtype triangle_dtype,
                           const core::Device &device)
    : Geometry(Geometry::GeometryType::TriangleMesh, 3),
      device_(device),
      vertex_attr_(TensorListMap("vertices")),
      triangle_attr_(TensorListMap("triangles")) {
    SetVertices(core::TensorList({3}, vertex_dtype, device_));
    SetTriangles(core::TensorList({3}, triangle_dtype, device_));
}

TriangleMesh::TriangleMesh(const core::TensorList &vertices,
                           const core::TensorList &triangles)
    : TriangleMesh(vertices.GetDtype(), triangles.GetDtype(), [&]() {
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

/// Create a TriangleMesh from a legacy Open3D TriangleMesh.
TriangleMesh TriangleMesh::FromLegacyTrangleMesh(
        const open3d::geometry::TriangleMesh &mesh_legacy,
        core::Dtype vertex_dtype,
        core::Dtype triangle_dtype,
        const core::Device &device) {
    TriangleMesh mesh(vertex_dtype, triangle_dtype, device);

    if (mesh_legacy.HasVertices()) {
        mesh.SetVertices(core::eigen_converter::EigenVector3dVectorToTensorList(
                mesh_legacy.vertices_, vertex_dtype, device));
    } else {
        utility::LogWarning(
                "Creating from an empty legacy pointcloud, an empty pointcloud "
                "with default dtype and device will be created.");
    }
    if (mesh_legacy.HasVertexColors()) {
        mesh.SetVertexColors(
                core::eigen_converter::EigenVector3dVectorToTensorList(
                        mesh_legacy.vertex_colors_, vertex_dtype, device));
    }
    if (mesh_legacy.HasVertexNormals()) {
        mesh.SetVertexNormals(
                core::eigen_converter::EigenVector3dVectorToTensorList(
                        mesh_legacy.vertex_normals_, vertex_dtype, device));
    }

    if (mesh_legacy.HasTriangles()) {
        mesh.SetTriangles(
                core::eigen_converter::EigenVector3iVectorToTensorList(
                        mesh_legacy.triangles_, triangle_dtype, device));
    }

    return mesh;
};

/// Convert to a legacy Open3D TriangleMesh.
open3d::geometry::TriangleMesh TriangleMesh::ToLegacyTriangleMesh() const {
    open3d::geometry::TriangleMesh mesh_legacy;
    if (HasVertices()) {
        core::Tensor vertices =
                GetVertices().AsTensor().Copy(core::Device("CPU:0"));
        int64_t N = vertices.GetShape()[0];
        mesh_legacy.vertices_.resize(N);
        for (int64_t i = 0; i < N; ++i) {
            mesh_legacy.vertices_[i] =
                    core::eigen_converter::TensorToEigenVector3d(vertices[i]);
        }
    }

    if (HasVertexNormals()) {
        core::Tensor normals =
                GetVertexNormals().AsTensor().Copy(core::Device("CPU:0"));
        int64_t N = normals.GetShape()[0];
        mesh_legacy.vertex_normals_.resize(N);
        for (int64_t i = 0; i < N; ++i) {
            mesh_legacy.vertex_normals_[i] =
                    core::eigen_converter::TensorToEigenVector3d(normals[i]);
        }
    }

    if (HasTriangles()) {
        core::Tensor triangles =
                GetTriangles().AsTensor().Copy(core::Device("CPU:0"));
        int64_t N = triangles.GetShape()[0];
        mesh_legacy.triangles_.resize(N);
        for (int64_t i = 0; i < N; ++i) {
            mesh_legacy.triangles_[i] =
                    core::eigen_converter::TensorToEigenVector3i(triangles[i]);
        }
    }

    return mesh_legacy;
};

}  // namespace geometry
}  // namespace t
}  // namespace open3d
