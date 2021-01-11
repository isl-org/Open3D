// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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

#include <string>
#include <unordered_map>

#include "pybind/t/geometry/geometry.h"

namespace open3d {
namespace t {
namespace geometry {

void pybind_trianglemesh(py::module& m) {
    py::class_<TriangleMesh, PyGeometry<TriangleMesh>,
               std::unique_ptr<TriangleMesh>, Geometry>
            triangle_mesh(
                    m, "TriangleMesh",
                    "A triangle mesh contains a set of 3d vertices and faces.");

    // Constructors.
    triangle_mesh.def(py::init<const core::Device&>(), "device"_a)
            .def(py::init<const core::Tensor&, const core::Tensor&>(),
                 "vertices"_a, "triangles"_a);

    // Triangle mesh's attributes: vertices, vertex_colors, vertex_normals, etc.
    // def_property_readonly is sufficient, since the returned TensorMap can
    // be editable in Python. We don't want the TensorMap to be replaced
    // by another TensorMap in Python.
    triangle_mesh.def_property_readonly(
            "vertices",
            py::overload_cast<>(&TriangleMesh::GetVertexAttr, py::const_));
    triangle_mesh.def_property_readonly(
            "triangles",
            py::overload_cast<>(&TriangleMesh::GetTriangleAttr, py::const_));

    // Device transfers.
    triangle_mesh.def("to", &TriangleMesh::To,
                      "Transfer the triangle mesh to a specified device.",
                      "device"_a, "copy"_a = false);
    triangle_mesh.def("clone", &TriangleMesh::Clone,
                      "Returns copy of the triangle mesh on the same device.");
    triangle_mesh.def("cpu", &TriangleMesh::CPU,
                      "Transfer the triangle mesh to CPU. If the triangle mesh "
                      "is already on CPU, no copy will be performed.");
    triangle_mesh.def(
            "cuda", &TriangleMesh::CUDA,
            "Transfer the triangle mesh to a CUDA device. If the triangle mesh "
            "is already on the specified CUDA device, no copy will be "
            "performed.",
            "device_id"_a = 0);

    // Triangle Mesh's specific functions.
    triangle_mesh.def("get_min_bound", &TriangleMesh::GetMinBound,
                      "Returns the min bound for point coordinates.");
    triangle_mesh.def("get_max_bound", &TriangleMesh::GetMaxBound,
                      "Returns the max bound for point coordinates.");
    triangle_mesh.def("get_center", &TriangleMesh::GetCenter,
                      "Returns the center for point coordinates.");
    triangle_mesh.def("transform", &TriangleMesh::Transform, "transformation"_a,
                      "Transforms the points and normals (if exist).");
    triangle_mesh.def("translate", &TriangleMesh::Translate, "translation"_a,
                      "relative"_a = true, "Translates points.");
    triangle_mesh.def("scale", &TriangleMesh::Scale, "scale"_a, "center"_a,
                      "Scale points.");
    triangle_mesh.def("rotate", &TriangleMesh::Rotate, "R"_a, "center"_a,
                      "Rotate points and normals (if exist).");
    triangle_mesh.def_static(
            "from_legacy_triangle_mesh", &TriangleMesh::FromLegacyTriangleMesh,
            "mesh_legacy"_a, "vertex_dtype"_a = core::Dtype::Float32,
            "triangle_dtype"_a = core::Dtype::Int64,
            "device"_a = core::Device("CPU:0"),
            "Create a TriangleMesh from a legacy Open3D TriangleMesh.");
    triangle_mesh.def("to_legacy_triangle_mesh",
                      &TriangleMesh::ToLegacyTriangleMesh,
                      "Convert to a legacy Open3D TriangleMesh.");
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
