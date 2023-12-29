// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/geometry/DrawableGeometry.h"

#include "pybind/t/geometry/geometry.h"

namespace open3d {
namespace t {
namespace geometry {

void pybind_drawable_geometry_class(py::module& m) {
    // open3d.t.geometry.DrawableGeometry
    py::class_<DrawableGeometry, std::shared_ptr<DrawableGeometry>>
            drawable_geometry(
                    m, "DrawableGeometry",
                    "Base class for geometry types which can be visualized.");
    drawable_geometry.def("has_valid_material", &DrawableGeometry::HasMaterial,
                          "Returns true if the geometry's material is valid.");
    drawable_geometry.def_property(
            "material", py::overload_cast<>(&DrawableGeometry::GetMaterial),
            &DrawableGeometry::SetMaterial);
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
