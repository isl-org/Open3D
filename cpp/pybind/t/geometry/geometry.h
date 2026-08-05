// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/core/Dtype.h"
#include "open3d/t/geometry/Geometry.h"
#include "pybind/open3d_pybind.h"

namespace open3d {
namespace t {
namespace geometry {

// Geometry trampoline class.
template <class GeometryBase = Geometry>
class PyGeometry : public GeometryBase {
public:
    using GeometryBase::GeometryBase;

    GeometryBase& Clear() override {
        PYBIND11_OVERLOAD_PURE(GeometryBase&, GeometryBase, );
    }

    bool IsEmpty() const override {
        PYBIND11_OVERLOAD_PURE(bool, GeometryBase, );
    }

    core::Device GetDevice() const override {
        PYBIND11_OVERLOAD_PURE(core::Device, GeometryBase, );
    }
};

void pybind_geometry_declarations(py::module& m);
void pybind_geometry_class_declarations(py::module& m);
void pybind_drawable_geometry_class_declarations(py::module& m);
void pybind_tensormap_declarations(py::module& m);
void pybind_pointcloud_declarations(py::module& m);
void pybind_lineset_declarations(py::module& m);
void pybind_trianglemesh_declarations(py::module& m);
void pybind_image_declarations(py::module& m);
void pybind_boundingvolume_declarations(py::module& m);
void pybind_voxel_block_grid_declarations(py::module& m);
void pybind_raycasting_scene_declarations(py::module& m);

void pybind_geometry_definitions(py::module& m);
void pybind_geometry_class_definitions(py::module& m);
void pybind_drawable_geometry_class_definitions(py::module& m);
void pybind_tensormap_definitions(py::module& m);
void pybind_pointcloud_definitions(py::module& m);
void pybind_lineset_definitions(py::module& m);
void pybind_trianglemesh_definitions(py::module& m);
void pybind_image_definitions(py::module& m);
void pybind_boundingvolume_definitions(py::module& m);
void pybind_voxel_block_grid_definitions(py::module& m);
void pybind_raycasting_scene_definitions(py::module& m);

}  // namespace geometry
}  // namespace t
}  // namespace open3d
