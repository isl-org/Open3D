// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/geometry/TetraMesh.h"

#include "open3d/geometry/PointCloud.h"
#include "pybind/docstring.h"
#include "pybind/geometry/geometry.h"
#include "pybind/geometry/geometry_trampoline.h"

namespace open3d {
namespace geometry {

void pybind_tetramesh(py::module &m) {
    py::class_<TetraMesh, PyGeometry3D<TetraMesh>, std::shared_ptr<TetraMesh>,
               MeshBase>
            trianglemesh(m, "TetraMesh",
                         "TetraMesh class. Tetra mesh contains vertices "
                         "and tetrahedra represented by the indices to the "
                         "vertices.");
    py::detail::bind_default_constructor<TetraMesh>(trianglemesh);
    py::detail::bind_copy_functions<TetraMesh>(trianglemesh);
    trianglemesh
            .def(py::init<const std::vector<Eigen::Vector3d> &,
                          const std::vector<Eigen::Vector4i,
                                            utility::Vector4i_allocator> &>(),
                 "Create a tetrahedra mesh from vertices and tetra indices",
                 "vertices"_a, "tetras"_a)
            .def("__repr__",
                 [](const TetraMesh &mesh) {
                     return std::string("TetraMesh with ") +
                            std::to_string(mesh.vertices_.size()) +
                            " points and " +
                            std::to_string(mesh.tetras_.size()) +
                            " tetrahedra.";
                 })
            .def(py::self + py::self)
            .def(py::self += py::self)
            .def("remove_duplicated_vertices",
                 &TetraMesh::RemoveDuplicatedVertices,
                 "Function that removes duplicated vertices, i.e., vertices "
                 "that have identical coordinates.")
            .def("remove_duplicated_tetras", &TetraMesh::RemoveDuplicatedTetras,
                 "Function that removes duplicated tetras, i.e., removes "
                 "tetras that reference the same four vertices, "
                 "independent of their order.")
            .def("remove_unreferenced_vertices",
                 &TetraMesh::RemoveUnreferencedVertices,
                 "This function removes vertices from the tetra mesh that "
                 "are not referenced in any tetra of the mesh.")
            .def("remove_degenerate_tetras", &TetraMesh::RemoveDegenerateTetras,
                 "Function that removes degenerate tetras, i.e., tetras "
                 "that references a single vertex multiple times in a single "
                 "tetra. They are usually the product of removing "
                 "duplicated vertices.")
            .def("has_vertices", &TetraMesh::HasVertices,
                 "Returns ``True`` if the mesh contains vertices.")
            .def("has_tetras", &TetraMesh::HasTetras,
                 "Returns ``True`` if the mesh contains tetras.")
            .def("extract_triangle_mesh", &TetraMesh::ExtractTriangleMesh,
                 "Function that generates a triangle mesh of the specified "
                 "iso-surface.",
                 "values"_a, "level"_a)
            .def_static(
                    "create_from_point_cloud", &TetraMesh::CreateFromPointCloud,
                    "Function to create a tetrahedral mesh from a point cloud.",
                    "point_cloud"_a)
            .def_readwrite("vertices", &TetraMesh::vertices_,
                           "``float64`` array of shape ``(num_vertices, 3)``, "
                           "use ``numpy.asarray()`` to access data: Vertex "
                           "coordinates.")
            .def_readwrite("tetras", &TetraMesh::tetras_,
                           "``int64`` array of shape ``(num_tetras, 4)``, use "
                           "``numpy.asarray()`` to access data: List of "
                           "tetras denoted by the index of points forming "
                           "the tetra.");
    docstring::ClassMethodDocInject(m, "TetraMesh", "has_tetras");
    docstring::ClassMethodDocInject(m, "TetraMesh", "has_vertices");
    docstring::ClassMethodDocInject(m, "TetraMesh",
                                    "remove_duplicated_vertices");
    docstring::ClassMethodDocInject(m, "TetraMesh", "remove_duplicated_tetras");
    docstring::ClassMethodDocInject(m, "TetraMesh",
                                    "remove_unreferenced_vertices");
    docstring::ClassMethodDocInject(m, "TetraMesh", "remove_degenerate_tetras");
    docstring::ClassMethodDocInject(
            m, "TetraMesh", "extract_triangle_mesh",
            {{"values",
              "Vector with a scalar value for each vertex in the tetra mesh"},
             {"level", "A scalar which defines the level-set to extract"}});
    docstring::ClassMethodDocInject(m, "TetraMesh", "create_from_point_cloud",
                                    {{"point_cloud", "A PointCloud."}});
}

void pybind_tetramesh_methods(py::module &m) {}

}  // namespace geometry
}  // namespace open3d
