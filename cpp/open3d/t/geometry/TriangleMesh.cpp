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

#include "libqhullcpp/PointCoordinates.h"
#include "libqhullcpp/Qhull.h"
#include "libqhullcpp/QhullFacet.h"
#include "libqhullcpp/QhullFacetList.h"
#include "libqhullcpp/QhullVertexSet.h"
#include "open3d/core/EigenConverter.h"
#include "open3d/core/ShapeUtil.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/TensorCheck.h"
#include "open3d/t/geometry/kernel/PointCloud.h"
#include "open3d/t/geometry/kernel/Transform.h"

namespace open3d {
namespace t {
namespace geometry {

TriangleMesh::TriangleMesh(const core::Device &device)
    : Geometry(Geometry::GeometryType::TriangleMesh, 3),
      device_(device),
      vertex_attr_(TensorMap("positions")),
      triangle_attr_(TensorMap("indices")) {}

TriangleMesh::TriangleMesh(const core::Tensor &vertex_positions,
                           const core::Tensor &triangle_indices)
    : TriangleMesh([&]() {
          if (vertex_positions.GetDevice() != triangle_indices.GetDevice()) {
              utility::LogError(
                      "vertex_positions' device {} does not match "
                      "triangle_indices' device {}.",
                      vertex_positions.GetDevice().ToString(),
                      triangle_indices.GetDevice().ToString());
          }
          return vertex_positions.GetDevice();
      }()) {
    SetVertexPositions(vertex_positions);
    SetTriangleIndices(triangle_indices);
}

std::string TriangleMesh::ToString() const {
    if (vertex_attr_.size() == 0 || triangle_attr_.size() == 0)
        return fmt::format("TriangleMesh on {} [{} vertices and {} triangles].",
                           GetDevice().ToString(), vertex_attr_.size(),
                           triangle_attr_.size());

    auto str = fmt::format(
            "TriangleMesh on {} [{} vertices ({}) and {} triangles ({})].",
            GetDevice().ToString(), GetVertexPositions().GetLength(),
            GetVertexPositions().GetDtype().ToString(),
            GetTriangleIndices().GetLength(),
            GetTriangleIndices().GetDtype().ToString());

    std::string vertices_attr_str = "\nVertex Attributes:";
    if (vertex_attr_.size() == 1) {
        vertices_attr_str += " None.";
    } else {
        for (const auto &kv : vertex_attr_) {
            if (kv.first != "positions") {
                vertices_attr_str +=
                        fmt::format(" {} (dtype = {}, shape = {}),", kv.first,
                                    kv.second.GetDtype().ToString(),
                                    kv.second.GetShape().ToString());
            }
        }
        vertices_attr_str[vertices_attr_str.size() - 1] = '.';
    }

    std::string triangles_attr_str = "\nTriangle Attributes:";
    if (triangle_attr_.size() == 1) {
        triangles_attr_str += " None.";
    } else {
        for (const auto &kv : triangle_attr_) {
            if (kv.first != "indices") {
                triangles_attr_str +=
                        fmt::format(" {} (dtype = {}, shape = {}),", kv.first,
                                    kv.second.GetDtype().ToString(),
                                    kv.second.GetShape().ToString());
            }
        }
        triangles_attr_str[triangles_attr_str.size() - 1] = '.';
    }

    return str + vertices_attr_str + triangles_attr_str;
}

TriangleMesh &TriangleMesh::Transform(const core::Tensor &transformation) {
    core::AssertTensorShape(transformation, {4, 4});

    kernel::transform::TransformPoints(transformation, GetVertexPositions());
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
    core::AssertTensorShape(translation, {3});

    core::Tensor transform =
            translation.To(GetDevice(), GetVertexPositions().GetDtype());

    if (!relative) {
        transform -= GetCenter();
    }
    GetVertexPositions() += transform;
    return *this;
}

TriangleMesh &TriangleMesh::Scale(double scale, const core::Tensor &center) {
    core::AssertTensorShape(center, {3});
    core::AssertTensorDevice(center, device_);

    const core::Tensor center_d =
            center.To(GetDevice(), GetVertexPositions().GetDtype());

    GetVertexPositions().Sub_(center_d).Mul_(scale).Add_(center_d);
    return *this;
}

TriangleMesh &TriangleMesh::Rotate(const core::Tensor &R,
                                   const core::Tensor &center) {
    core::AssertTensorShape(R, {3, 3});
    core::AssertTensorShape(center, {3});

    kernel::transform::RotatePoints(R, GetVertexPositions(), center);
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
        mesh.SetVertexPositions(
                core::eigen_converter::EigenVector3dVectorToTensor(
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
        mesh.SetTriangleIndices(
                core::eigen_converter::EigenVector3iVectorToTensor(
                        mesh_legacy.triangles_, int_dtype, device));
    }
    if (mesh_legacy.HasTriangleNormals()) {
        mesh.SetTriangleNormals(
                core::eigen_converter::EigenVector3dVectorToTensor(
                        mesh_legacy.triangle_normals_, float_dtype, device));
    }
    if (mesh_legacy.HasTriangleUvs()) {
        mesh.SetTriangleAttr(
                "texture_uvs",
                core::eigen_converter::EigenVector2dVectorToTensor(
                        mesh_legacy.triangle_uvs_, float_dtype, device)
                        .Reshape({-1, 3, 2}));
    }
    return mesh;
}

open3d::geometry::TriangleMesh TriangleMesh::ToLegacy() const {
    open3d::geometry::TriangleMesh mesh_legacy;
    if (HasVertexPositions()) {
        mesh_legacy.vertices_ =
                core::eigen_converter::TensorToEigenVector3dVector(
                        GetVertexPositions());
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
    if (HasTriangleIndices()) {
        mesh_legacy.triangles_ =
                core::eigen_converter::TensorToEigenVector3iVector(
                        GetTriangleIndices());
    }
    if (HasTriangleNormals()) {
        mesh_legacy.triangle_normals_ =
                core::eigen_converter::TensorToEigenVector3dVector(
                        GetTriangleNormals());
    }
    if (HasTriangleAttr("texture_uvs")) {
        mesh_legacy.triangle_uvs_ =
                core::eigen_converter::TensorToEigenVector2dVector(
                        GetTriangleAttr("texture_uvs").Reshape({-1, 2}));
    }
    if (HasVertexAttr("texture_uvs")) {
        utility::LogWarning("{}",
                            "texture_uvs as a vertex attribute is not "
                            "supported by legacy TriangleMesh. Ignored.");
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

TriangleMesh TriangleMesh::ComputeConvexHull(bool joggle_inputs) const {
    // QHull needs double dtype on the CPU.
    static_assert(std::is_same<realT, double>::value,
                  "QHull realT is not double. Update code!");
    using int_t = int32_t;
    auto float_dtype = GetVertexPositions().GetDtype();
    core::Tensor coordinates(
            GetVertexPositions().To(core::Float64).To(core::Device("CPU:0")));

    orgQhull::Qhull qhull;
    std::string options = "Qt";
    if (joggle_inputs) {
        options += " QJ";
    }
    qhull.runQhull("", 3, coordinates.GetShape(1),
                   coordinates.GetDataPtr<double>(), options.c_str());
    orgQhull::QhullFacetList facets = qhull.facetList();

    int_t n_good_facets = 0;
    for (orgQhull::QhullFacetList::iterator it = facets.begin();
         it != facets.end(); ++it) {
        if ((*it).isGood()) ++n_good_facets;
    }

    core::Tensor vertices({qhull.vertexCount(), 3}, float_dtype),
            triangles({n_good_facets, 3}, core::Int64),
            point_index({qhull.vertexCount()}, core::Int64);
    std::unordered_map<int_t, int_t> vertex_map;  // pcd -> conv hull
    int_t tidx = 0, next_vtx = 0;
    for (orgQhull::QhullFacetList::iterator it = facets.begin();
         it != facets.end(); ++it) {
        if (!(*it).isGood()) continue;

        orgQhull::QhullVertexSet vSet = it->vertices();
        int_t triangle_subscript = 0;
        for (orgQhull::QhullVertexSet::iterator vIt = vSet.begin();
             vIt != vSet.end(); ++vIt) {
            orgQhull::QhullPoint p = (*vIt).point();
            int_t vidx = p.id();
            triangles[tidx][triangle_subscript++] = vidx;  // pcd vertex idx

            if (vertex_map.count(vidx) == 0) {
                vertex_map[vidx] = int_t(next_vtx);
                double *coords = p.coordinates();
                if (float_dtype == core::Float64) {
                    double *pt = vertices[next_vtx].GetDataPtr<double>();
                    pt[0] = coords[0];
                    pt[1] = coords[1];
                    pt[2] = coords[2];
                } else {
                    float *pt = vertices[next_vtx].GetDataPtr<float>();
                    pt[0] = static_cast<float>(coords[0]);
                    pt[1] = static_cast<float>(coords[1]);
                    pt[2] = static_cast<float>(coords[2]);
                }
                point_index[next_vtx++] = vidx;
            }
        }
        tidx++;
    }
    if (next_vtx != vertices.GetShape(0)) {
        utility::LogError(
                "QHull output has incorrect number of vertices {} instead of "
                "reported {}",
                next_vtx, vertices.GetShape(0));
    }

    auto center = vertices.Mean({0});
    core::Tensor face_center({3, 3}, vertices.GetDtype());
    for (int_t k = 0; k < triangles.GetShape(0); ++k) {
        auto triangle = triangles[k].GetDataPtr<int_t>();
        // switch from pcd vertex idx to convex hull vertex idx
        triangle[0] = vertex_map[triangle[0]];
        triangle[1] = vertex_map[triangle[1]];
        triangle[2] = vertex_map[triangle[2]];

        // orient triangle vertices so that normal points out
        auto triangle_center =
                (1. / 3) * (vertices[triangle[0]] + vertices[triangle[1]] +
                            vertices[triangle[2]]);
        face_center[0] = triangle_center - center;
        face_center[1] = vertices[triangle[1]] - vertices[triangle[0]];
        face_center[2] = vertices[triangle[2]] - vertices[triangle[0]];
        if (face_center.Det() < 0) {
            std::swap(triangle[0], triangle[1]);
        }
    }

    TriangleMesh convex_hull(vertices, triangles);
    convex_hull.SetVertexAttr("point_index", point_index);
    return convex_hull;
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
