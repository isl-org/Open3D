// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/geometry/MeshBase.h"

#include "open3d/geometry/PointCloud.h"
#include "pybind/docstring.h"
#include "pybind/geometry/geometry.h"
#include "pybind/geometry/geometry_trampoline.h"

namespace open3d {
namespace geometry {

void pybind_meshbase(py::module &m) {
    py::class_<MeshBase, PyGeometry3D<MeshBase>, std::shared_ptr<MeshBase>,
               Geometry3D>
            meshbase(m, "MeshBase",
                     "MeshBase class. Triangle mesh contains vertices. "
                     "Optionally, the mesh "
                     "may also contain vertex normals and vertex colors.");
    py::detail::bind_default_constructor<MeshBase>(meshbase);
    py::detail::bind_copy_functions<MeshBase>(meshbase);

    py::enum_<MeshBase::SimplificationContraction>(m,
                                                   "SimplificationContraction")
            .value("Average", MeshBase::SimplificationContraction::Average,
                   "The vertex positions are computed by the averaging.")
            .value("Quadric", MeshBase::SimplificationContraction::Quadric,
                   "The vertex positions are computed by minimizing the "
                   "distance to the adjacent triangle planes.")
            .export_values();

    py::enum_<MeshBase::FilterScope>(m, "FilterScope")
            .value("All", MeshBase::FilterScope::All,
                   "All properties (color, normal, vertex position) are "
                   "filtered.")
            .value("Color", MeshBase::FilterScope::Color,
                   "Only the color values are filtered.")
            .value("Normal", MeshBase::FilterScope::Normal,
                   "Only the normal values are filtered.")
            .value("Vertex", MeshBase::FilterScope::Vertex,
                   "Only the vertex positions are filtered.")
            .export_values();

    py::enum_<MeshBase::DeformAsRigidAsPossibleEnergy>(
            m, "DeformAsRigidAsPossibleEnergy")
            .value("Spokes", MeshBase::DeformAsRigidAsPossibleEnergy::Spokes,
                   "is the original energy as formulated in orkine and Alexa, "
                   "\"As-Rigid-As-Possible Surface Modeling\", 2007.")
            .value("Smoothed",
                   MeshBase::DeformAsRigidAsPossibleEnergy::Smoothed,
                   "adds a rotation smoothing term to the rotations.")
            .export_values();

    meshbase.def("__repr__",
                 [](const MeshBase &mesh) {
                     return std::string("MeshBase with ") +
                            std::to_string(mesh.vertices_.size()) + " points";
                 })
            .def(py::self + py::self)
            .def(py::self += py::self)
            .def("has_vertices", &MeshBase::HasVertices,
                 "Returns ``True`` if the mesh contains vertices.")
            .def("has_vertex_normals", &MeshBase::HasVertexNormals,
                 "Returns ``True`` if the mesh contains vertex normals.")
            .def("has_vertex_colors", &MeshBase::HasVertexColors,
                 "Returns ``True`` if the mesh contains vertex colors.")
            .def("normalize_normals", &MeshBase::NormalizeNormals,
                 "Normalize vertex normals to length 1.")
            .def("paint_uniform_color", &MeshBase::PaintUniformColor,
                 "Assigns each vertex in the MeshBase the same color.",
                 "color"_a)
            .def("compute_convex_hull", &MeshBase::ComputeConvexHull,
                 "Computes the convex hull of the triangle mesh.")
            .def_readwrite("vertices", &MeshBase::vertices_,
                           "``float64`` array of shape ``(num_vertices, 3)``, "
                           "use ``numpy.asarray()`` to access data: Vertex "
                           "coordinates.")
            .def_readwrite("vertex_normals", &MeshBase::vertex_normals_,
                           "``float64`` array of shape ``(num_vertices, 3)``, "
                           "use ``numpy.asarray()`` to access data: Vertex "
                           "normals.")
            .def_readwrite(
                    "vertex_colors", &MeshBase::vertex_colors_,
                    "``float64`` array of shape ``(num_vertices, 3)``, "
                    "range ``[0, 1]`` , use ``numpy.asarray()`` to access "
                    "data: RGB colors of vertices.");
    docstring::ClassMethodDocInject(m, "MeshBase", "has_vertex_colors");
    docstring::ClassMethodDocInject(
            m, "MeshBase", "has_vertex_normals",
            {{"normalized",
              "Set to ``True`` to normalize the normal to length 1."}});
    docstring::ClassMethodDocInject(m, "MeshBase", "has_vertices");
    docstring::ClassMethodDocInject(m, "MeshBase", "normalize_normals");
    docstring::ClassMethodDocInject(m, "MeshBase", "paint_uniform_color",
                                    {{"color", "RGB colors of vertices."}});
    docstring::ClassMethodDocInject(m, "MeshBase", "compute_convex_hull");
}

void pybind_meshbase_methods(py::module &m) {}

}  // namespace geometry
}  // namespace open3d
