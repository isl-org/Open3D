// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "pybind/open3d_pybind.h"

namespace open3d {
namespace t {
namespace io {

void pybind_io_declarations(py::module& m);
void pybind_class_io_declarations(py::module& m);
void pybind_sensor_declarations(py::module& m);

void pybind_io_definitions(py::module& m);
void pybind_class_io_definitions(py::module& m);
void pybind_sensor_definitions(py::module& m);

}  // namespace io
}  // namespace t
}  // namespace open3d
