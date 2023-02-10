// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/geometry/Geometry.h"

#include "pybind/docstring.h"
#include "pybind/t/geometry/geometry.h"

namespace open3d {
namespace t {
namespace geometry {

void pybind_geometry_class(py::module& m) {
    // open3d.t.geometry.Geometry
    py::class_<Geometry, PyGeometry<Geometry>, std::shared_ptr<Geometry>>
            geometry(m, "Geometry", "The base geometry class.");

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

void pybind_geometry(py::module& m) {
    py::module m_submodule = m.def_submodule(
            "geometry", "Tensor-based geometry defining module.");

    pybind_geometry_class(m_submodule);
    pybind_drawable_geometry_class(m_submodule);
    pybind_tensormap(m_submodule);
    pybind_pointcloud(m_submodule);
    pybind_lineset(m_submodule);
    pybind_trianglemesh(m_submodule);
    pybind_image(m_submodule);
    pybind_boundingvolume(m_submodule);
    pybind_voxel_block_grid(m_submodule);
    pybind_raycasting_scene(m_submodule);
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
