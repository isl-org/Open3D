// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/t/t.h"

#include "pybind/open3d_pybind.h"
#include "pybind/t/geometry/geometry.h"
#include "pybind/t/io/io.h"
#include "pybind/t/pipelines/pipelines.h"

namespace open3d {
namespace t {

void pybind_t_declarations(py::module& m) {
    py::module m_t = m.def_submodule("t");
    pipelines::pybind_pipelines_declarations(m_t);
    geometry::pybind_geometry_declarations(m_t);
    io::pybind_io_declarations(m_t);
}
void pybind_t_definitions(py::module& m) {
    auto m_t = static_cast<py::module>(m.attr("t"));
    pipelines::pybind_pipelines_definitions(m_t);
    geometry::pybind_geometry_definitions(m_t);
    io::pybind_io_definitions(m_t);
}

}  // namespace t
}  // namespace open3d
