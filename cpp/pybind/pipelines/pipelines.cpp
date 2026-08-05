// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/pipelines/pipelines.h"

#include "pybind/open3d_pybind.h"
#include "pybind/pipelines/color_map/color_map.h"
#include "pybind/pipelines/integration/integration.h"
#include "pybind/pipelines/odometry/odometry.h"
#include "pybind/pipelines/registration/registration.h"

namespace open3d {
namespace pipelines {

void pybind_pipelines_declarations(py::module& m) {
    py::module m_pipelines = m.def_submodule("pipelines");
    color_map::pybind_color_map_declarations(m_pipelines);
    integration::pybind_integration_declarations(m_pipelines);
    registration::pybind_registration_declarations(m_pipelines);
    odometry::pybind_odometry_declarations(m_pipelines);
}

void pybind_pipelines_definitions(py::module& m) {
    auto m_pipelines = static_cast<py::module>(m.attr("pipelines"));
    color_map::pybind_color_map_definitions(m_pipelines);
    integration::pybind_integration_definitions(m_pipelines);
    registration::pybind_registration_definitions(m_pipelines);
    odometry::pybind_odometry_definitions(m_pipelines);
}

}  // namespace pipelines
}  // namespace open3d
