// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/geometry/Geometry.h"

#include "pybind/docstring.h"
#include "pybind/t/geometry/geometry.h"

namespace open3d {
namespace t {
namespace geometry {

void pybind_geometry_class_declarations(py::module& m) {
    py::class_<Geometry, PyGeometry<Geometry>, std::shared_ptr<Geometry>>
            geometry(m, "Geometry", "The base geometry class.");
}

void pybind_geometry_class_definitions(py::module& m) {
    // open3d.t.geometry.Geometry
    auto geometry = static_cast<py::class_<Geometry, PyGeometry<Geometry>,
                                           std::shared_ptr<Geometry>>>(
            m.attr("Geometry"));
    geometry.def("clear", &Geometry::Clear,
                 "Clear all elements in the geometry.");
    geometry.def("is_empty", &Geometry::IsEmpty,
                 "Returns ``True`` iff the geometry is empty.");
    geometry.def_property_readonly("device", &Geometry::GetDevice,
                                   "Returns the device of the geometry.");
    geometry.def_property_readonly("is_cpu", &Geometry::IsCPU,
                                   "Returns true if the geometry is on CPU.");
    geometry.def_property_readonly("is_cuda", &Geometry::IsCUDA,
                                   "Returns true if the geometry is on CUDA.");
    docstring::ClassMethodDocInject(m, "Geometry", "clear");
    docstring::ClassMethodDocInject(m, "Geometry", "is_empty");
}

void pybind_geometry_declarations(py::module& m) {
    py::module m_geometry = m.def_submodule(
            "geometry", "Tensor-based geometry defining module.");

    pybind_geometry_class_declarations(m_geometry);
    pybind_drawable_geometry_class_declarations(m_geometry);
    pybind_tensormap_declarations(m_geometry);
    pybind_pointcloud_declarations(m_geometry);
    pybind_lineset_declarations(m_geometry);
    pybind_trianglemesh_declarations(m_geometry);
    pybind_image_declarations(m_geometry);
    pybind_boundingvolume_declarations(m_geometry);
    pybind_voxel_block_grid_declarations(m_geometry);
    pybind_raycasting_scene_declarations(m_geometry);
}

void pybind_geometry_definitions(py::module& m) {
    auto m_geometry = static_cast<py::module>(m.attr("geometry"));
    pybind_geometry_class_definitions(m_geometry);
    pybind_drawable_geometry_class_definitions(m_geometry);
    pybind_tensormap_definitions(m_geometry);
    pybind_pointcloud_definitions(m_geometry);
    pybind_lineset_definitions(m_geometry);
    pybind_trianglemesh_definitions(m_geometry);
    pybind_image_definitions(m_geometry);
    pybind_boundingvolume_definitions(m_geometry);
    pybind_voxel_block_grid_definitions(m_geometry);
    pybind_raycasting_scene_definitions(m_geometry);
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
