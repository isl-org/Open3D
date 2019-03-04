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

#include "Python/geometry/geometry_trampoline.h"
#include "Python/geometry/geometry.h"

#include <Open3D/Geometry/HalfEdgeTriangleMesh.h>
#include <sstream>

using namespace open3d;

void pybind_half_edge(py::module &m) {
    py::class_<geometry::HalfEdgeTriangleMesh::HalfEdge> half_edge(
            m, "HalfEdge", "HalfEdge");
    py::detail::bind_default_constructor<
            geometry::HalfEdgeTriangleMesh::HalfEdge>(half_edge);
    py::detail::bind_copy_functions<geometry::HalfEdgeTriangleMesh::HalfEdge>(
            half_edge);
    half_edge
            .def("__repr__",
                 [](const geometry::HalfEdgeTriangleMesh::HalfEdge &he) {
                     std::ostringstream repr;
                     repr << "HalfEdge(vertex_indices {"
                          << he.vertex_indices_(0) << ", "
                          << he.vertex_indices_(1) << "}, triangle_index "
                          << he.triangle_index_ << ", next " << he.next_
                          << ", twin " << he.twin_ << ")";
                     return repr.str();
                 })
            .def("is_boundary",
                 &geometry::HalfEdgeTriangleMesh::HalfEdge::IsBoundary)
            .def_readwrite("next",
                           &geometry::HalfEdgeTriangleMesh::HalfEdge::next_)
            .def_readwrite("twin",
                           &geometry::HalfEdgeTriangleMesh::HalfEdge::twin_)
            .def_readwrite(
                    "vertex_indices",
                    &geometry::HalfEdgeTriangleMesh::HalfEdge::vertex_indices_)
            .def_readwrite(
                    "triangle_index",
                    &geometry::HalfEdgeTriangleMesh::HalfEdge::triangle_index_);
}

void pybind_halfedgetrianglemesh(py::module &m) {
    pybind_half_edge(m);

    py::class_<geometry::HalfEdgeTriangleMesh,
               PyTriangleMesh<geometry::HalfEdgeTriangleMesh>,
               std::shared_ptr<geometry::HalfEdgeTriangleMesh>,
               geometry::TriangleMesh>
            half_edge_triangle_mesh(m, "HalfEdgeTriangleMesh",
                                    "HalfEdgeTriangleMesh");
    py::detail::bind_default_constructor<geometry::HalfEdgeTriangleMesh>(
            half_edge_triangle_mesh);
    py::detail::bind_copy_functions<geometry::HalfEdgeTriangleMesh>(
            half_edge_triangle_mesh);
    half_edge_triangle_mesh
            .def("__repr__",
                 [](const geometry::HalfEdgeTriangleMesh &mesh) {
                     return std::string(
                                    "geometry::HalfEdgeTriangleMesh with ") +
                            std::to_string(mesh.vertices_.size()) +
                            " points and " +
                            std::to_string(mesh.triangles_.size()) +
                            " triangles.";
                 })
            .def("compute_half_edges",
                 &geometry::HalfEdgeTriangleMesh::ComputeHalfEdges)
            .def("has_half_edges",
                 &geometry::HalfEdgeTriangleMesh::HasHalfEdges)
            .def("boundary_half_edges_from_vertex",
                 &geometry::HalfEdgeTriangleMesh::BoundaryHalfEdgesFromVertex,
                 "vertex_index"_a)
            .def("boundary_vertices_from_vertex",
                 &geometry::HalfEdgeTriangleMesh::BoundaryVerticesFromVertex,
                 "vertex_index"_a)
            .def("get_boundaries",
                 &geometry::HalfEdgeTriangleMesh::GetBoundaries)
            .def_readwrite("half_edges",
                           &geometry::HalfEdgeTriangleMesh::half_edges_)
            .def_readwrite("ordered_half_edge_from_vertex",
                           &geometry::HalfEdgeTriangleMesh::
                                   ordered_half_edge_from_vertex_);

    m.def("create_half_edge_mesh_from_mesh",
          &geometry::CreateHalfEdgeMeshFromMesh, "mesh"_a);
}
