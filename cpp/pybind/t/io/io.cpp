// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/t/io/io.h"

#include "pybind/open3d_pybind.h"

namespace open3d {
namespace t {
namespace io {

void pybind_io_declarations(py::module& m) {
    py::module m_io =
            m.def_submodule("io", "Tensor-based input-output handling module.");
    pybind_class_io_declarations(m_io);
    pybind_sensor_declarations(m_io);
}
void pybind_io_definitions(py::module& m) {
    auto m_io = static_cast<py::module>(m.attr("io"));
    pybind_class_io_definitions(m_io);
    pybind_sensor_definitions(m_io);
}

}  // namespace io
}  // namespace t
}  // namespace open3d
