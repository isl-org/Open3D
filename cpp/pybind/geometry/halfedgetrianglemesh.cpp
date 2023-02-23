// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/geometry/HalfEdgeTriangleMesh.h"

#include <sstream>

#include "pybind/docstring.h"
#include "pybind/geometry/geometry.h"
#include "pybind/geometry/geometry_trampoline.h"

namespace open3d {
namespace geometry {

void pybind_half_edge(py::module &m) {
    py::class_<HalfEdgeTriangleMesh::HalfEdge> half_edge(
            m, "HalfEdge",
            "HalfEdge class contains vertex, triangle info about a half edge, "
            "as well as relations of next and twin half edge.");
    py::detail::bind_default_constructor<HalfEdgeTriangleMesh::HalfEdge>(
            half_edge);
    py::detail::bind_copy_functions<HalfEdgeTriangleMesh::HalfEdge>(half_edge);
    half_edge
            .def("__repr__",
                 [](const HalfEdgeTriangleMesh::HalfEdge &he) {
                     std::ostringstream repr;
                     repr << "HalfEdge(vertex_indices {"
                          << he.vertex_indices_(0) << ", "
                          << he.vertex_indices_(1) << "}, triangle_index "
                          << he.triangle_index_ << ", next " << he.next_
                          << ", twin " << he.twin_ << ")";
                     return repr.str();
                 })
            .def("is_boundary", &HalfEdgeTriangleMesh::HalfEdge::IsBoundary,
                 "Returns ``True`` iff the half edge is the boundary (has not "
                 "twin, i.e. twin index == -1).")
            .def_readwrite(
                    "next", &HalfEdgeTriangleMesh::HalfEdge::next_,
                    "int: Index of the next HalfEdge in the same triangle.")
            .def_readwrite("twin", &HalfEdgeTriangleMesh::HalfEdge::twin_,
                           "int: Index of the twin HalfEdge")
            .def_readwrite(
                    "vertex_indices",
                    &HalfEdgeTriangleMesh::HalfEdge::vertex_indices_,
                    "List(int) of length 2: Index of the ordered vertices "
                    "forming this half edge")
            .def_readwrite(
                    "triangle_index",
                    &HalfEdgeTriangleMesh::HalfEdge::triangle_index_,
                    "int: Index of the triangle containing this half edge");
}

void pybind_halfedgetrianglemesh(py::module &m) {
    pybind_half_edge(m);

    // open3d.geometry.HalfEdgeTriangleMesh
    py::class_<HalfEdgeTriangleMesh, PyGeometry3D<HalfEdgeTriangleMesh>,
               std::shared_ptr<HalfEdgeTriangleMesh>, MeshBase>
            half_edge_triangle_mesh(
                    m, "HalfEdgeTriangleMesh",
                    "HalfEdgeTriangleMesh inherits TriangleMesh class with the "
                    "addition of HalfEdge data structure for each half edge in "
                    "the mesh as well as related functions.");
    py::detail::bind_default_constructor<HalfEdgeTriangleMesh>(
            half_edge_triangle_mesh);
    py::detail::bind_copy_functions<HalfEdgeTriangleMesh>(
            half_edge_triangle_mesh);
    half_edge_triangle_mesh
            .def("__repr__",
                 [](const HalfEdgeTriangleMesh &mesh) {
                     return std::string("HalfEdgeTriangleMesh with ") +
                            std::to_string(mesh.vertices_.size()) +
                            " points and " +
                            std::to_string(mesh.half_edges_.size()) +
                            " half edges.";
                 })
            .def_readwrite("triangles", &HalfEdgeTriangleMesh::triangles_,
                           "``int`` array of shape ``(num_triangles, 3)``, use "
                           "``numpy.asarray()`` to access data: List of "
                           "triangles denoted by the index of points forming "
                           "the triangle.")
            .def_readwrite("triangle_normals",
                           &HalfEdgeTriangleMesh::triangle_normals_,
                           "``float64`` array of shape ``(num_triangles, 3)``, "
                           "use ``numpy.asarray()`` to access data: Triangle "
                           "normals.")
            .def("has_half_edges", &HalfEdgeTriangleMesh::HasHalfEdges,
                 "Returns ``True`` if half-edges have already been computed.")
            .def("boundary_half_edges_from_vertex",
                 &HalfEdgeTriangleMesh::BoundaryHalfEdgesFromVertex,
                 "vertex_index"_a,
                 "Query manifold boundary half edges from a starting vertex. "
                 "If query vertex is not on boundary, empty vector will be "
                 "returned.")
            .def("boundary_vertices_from_vertex",
                 &HalfEdgeTriangleMesh::BoundaryVerticesFromVertex,
                 "vertex_index"_a
                 "Query manifold boundary vertices from a starting vertex. If "
                 "query vertex is not on boundary, empty vector will be "
                 "returned.")
            .def("get_boundaries", &HalfEdgeTriangleMesh::GetBoundaries,
                 "Returns a vector of boundaries. A boundary is a vector of "
                 "vertices.")
            .def_static("create_from_triangle_mesh",
                        &HalfEdgeTriangleMesh::CreateFromTriangleMesh, "mesh"_a,
                        "Convert HalfEdgeTriangleMesh from TriangleMesh. "
                        "Throws exception if "
                        "the input mesh is not manifolds")
            .def_readwrite("half_edges", &HalfEdgeTriangleMesh::half_edges_,
                           "List of HalfEdge in the mesh")
            .def_readwrite(
                    "ordered_half_edge_from_vertex",
                    &HalfEdgeTriangleMesh::ordered_half_edge_from_vertex_,
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

}  // namespace geometry
}  // namespace open3d
