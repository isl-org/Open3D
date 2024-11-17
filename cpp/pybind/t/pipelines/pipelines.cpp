// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/t/pipelines/pipelines.h"

#include "pybind/open3d_pybind.h"
#include "pybind/t/pipelines/odometry/odometry.h"
#include "pybind/t/pipelines/registration/registration.h"
#include "pybind/t/pipelines/slac/slac.h"
#include "pybind/t/pipelines/slam/slam.h"

namespace open3d {
namespace t {
namespace pipelines {

void pybind_pipelines_declarations(py::module& m) {
    py::module m_pipelines = m.def_submodule(
            "pipelines", "Tensor-based geometry processing pipelines.");
    odometry::pybind_odometry_declarations(m_pipelines);
    registration::pybind_registration_declarations(m_pipelines);
    slac::pybind_slac_declarations(m_pipelines);
    slam::pybind_slam_declarations(m_pipelines);
}
void pybind_pipelines_definitions(py::module& m) {
    auto m_pipelines = static_cast<py::module>(m.attr("pipelines"));
    odometry::pybind_odometry_definitions(m_pipelines);
    registration::pybind_registration_definitions(m_pipelines);
    slac::pybind_slac_definitions(m_pipelines);
    slam::pybind_slam_definitions(m_pipelines);
}

}  // namespace pipelines
}  // namespace t
}  // namespace open3d
