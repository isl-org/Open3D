// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
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

void pybind_geometry(py::module& m);
void pybind_geometry_class(py::module& m);
void pybind_drawable_geometry_class(py::module& m);
void pybind_tensormap(py::module& m);
void pybind_image(py::module& m);
void pybind_pointcloud(py::module& m);
void pybind_lineset(py::module& m);
void pybind_trianglemesh(py::module& m);
void pybind_image(py::module& m);
void pybind_boundingvolume(py::module& m);
void pybind_voxel_block_grid(py::module& m);
void pybind_raycasting_scene(py::module& m);

}  // namespace geometry
}  // namespace t
}  // namespace open3d
