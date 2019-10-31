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

#include <sstream>

#include "Open3D/Geometry/HalfEdgeTriangleMesh.h"

#include "open3d_pybind/docstring.h"
#include "open3d_pybind/geometry/geometry.h"
#include "open3d_pybind/geometry/geometry_trampoline.h"

using namespace open3d;

void pybind_half_edge(py::module &m) {
    py::class_<geometry::HalfEdgeTriangleMesh::HalfEdge> half_edge(
            m, "HalfEdge",
            "HalfEdge class contains vertex, triangle info about a half edge, "
            "as well as relations of next and twin half edge.");
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
                 &geometry::HalfEdgeTriangleMesh::HalfEdge::IsBoundary,
                 "Returns ``True`` iff the half edge is the boundary (has not "
                 "twin, i.e. twin index == -1).")
            .def_readwrite(
                    "next", &geometry::HalfEdgeTriangleMesh::HalfEdge::next_,
                    "int: Index of the next HalfEdge in the same triangle.")
            .def_readwrite("twin",
                           &geometry::HalfEdgeTriangleMesh::HalfEdge::twin_,
                           "int: Index of the twin HalfEdge")
            .def_readwrite(
                    "vertex_indices",
                    &geometry::HalfEdgeTriangleMesh::HalfEdge::vertex_indices_,
                    "List(int) of length 2: Index of the ordered vertices "
                    "forming this half edge")
            .def_readwrite(
                    "triangle_index",
                    &geometry::HalfEdgeTriangleMesh::HalfEdge::triangle_index_,
                    "int: Index of the triangle containing this half edge");
}

void pybind_halfedgetrianglemesh(py::module &m) {
    pybind_half_edge(m);

    // open3d.geometry.HalfEdgeTriangleMesh
    py::class_<geometry::HalfEdgeTriangleMesh,
               PyGeometry3D<geometry::HalfEdgeTriangleMesh>,
               std::shared_ptr<geometry::HalfEdgeTriangleMesh>,
               geometry::MeshBase>
            half_edge_triangle_mesh(
                    m, "HalfEdgeTriangleMesh",
                    "HalfEdgeTriangleMesh inherits TriangleMesh class with the "
                    "addition of HalfEdge data structure for each half edge in "
                    "the mesh as well as related functions.");
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
                            std::to_string(mesh.half_edges_.size()) +
                            " half edges.";
                 })
            .def_readwrite("triangles",
                           &geometry::HalfEdgeTriangleMesh::triangles_,
                           "``int`` array of shape ``(num_triangles, 3)``, use "
                           "``numpy.asarray()`` to access data: List of "
                           "triangles denoted by the index of points forming "
                           "the triangle.")
            .def_readwrite("triangle_normals",
                           &geometry::HalfEdgeTriangleMesh::triangle_normals_,
                           "``float64`` array of shape ``(num_triangles, 3)``, "
                           "use ``numpy.asarray()`` to access data: Triangle "
                           "normals.")
            .def("has_half_edges",
                 &geometry::HalfEdgeTriangleMesh::HasHalfEdges,
                 "Returns ``True`` if half-edges have already been computed.")
            .def("boundary_half_edges_from_vertex",
                 &geometry::HalfEdgeTriangleMesh::BoundaryHalfEdgesFromVertex,
                 "vertex_index"_a,
                 "Query manifold boundary half edges from a starting vertex. "
                 "If query vertex is not on boundary, empty vector will be "
                 "returned.")
            .def("boundary_vertices_from_vertex",
                 &geometry::HalfEdgeTriangleMesh::BoundaryVerticesFromVertex,
                 "vertex_index"_a
                 "Query manifold boundary vertices from a starting vertex. If "
                 "query vertex is not on boundary, empty vector will be "
                 "returned.")
            .def("get_boundaries",
                 &geometry::HalfEdgeTriangleMesh::GetBoundaries,
                 "Returns a vector of boundaries. A boundary is a vector of "
                 "vertices.")
            .def_static("create_from_triangle_mesh",
                        &geometry::HalfEdgeTriangleMesh::CreateFromTriangleMesh,
                        "mesh"_a,
                        "Convert HalfEdgeTriangleMesh from TriangleMesh. "
                        "Throws exception if "
                        "the input mesh is not manifolds")
            .def_readwrite("half_edges",
                           &geometry::HalfEdgeTriangleMesh::half_edges_,
                           "List of HalfEdge in the mesh")
            .def_readwrite("ordered_half_edge_from_vertex",
                           &geometry::HalfEdgeTriangleMesh::
                                   ordered_half_edge_from_vertex_,
                           "Counter-clockwise ordered half-edges started from "
                           "each vertex");
    docstring::ClassMethodDocInject(m, "HalfEdgeTriangleMesh",
                                    "boundary_half_edges_from_vertex");
    docstring::ClassMethodDocInject(m, "HalfEdgeTriangleMesh",
                                    "boundary_vertices_from_vertex");
    docstring::ClassMethodDocInject(m, "HalfEdgeTriangleMesh",
                                    "get_boundaries");
    docstring::ClassMethodDocInject(m, "HalfEdgeTriangleMesh",
                                    "has_half_edges");
    docstring::ClassMethodDocInject(m, "HalfEdgeTriangleMesh",
                                    "create_from_triangle_mesh",
                                    {{"mesh", "The input TriangleMesh"}});
}
