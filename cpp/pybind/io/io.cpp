// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/io/io.h"

#include "pybind/open3d_pybind.h"

namespace open3d {
namespace io {

void pybind_io_declarations(py::module &m) {
    py::module m_io = m.def_submodule("io");
    pybind_class_io_declarations(m_io);
    pybind_rpc_declarations(m_io);
#ifdef BUILD_AZURE_KINECT
    pybind_sensor_declarations(m_io);
#endif
}

void pybind_io_definitions(py::module &m) {
    auto m_io = static_cast<py::module>(m.attr("io"));
    pybind_class_io_definitions(m_io);
    pybind_rpc_definitions(m_io);
#ifdef BUILD_AZURE_KINECT
    pybind_sensor_definitions(m_io);
#endif
}

}  // namespace io
}  // namespace open3d
