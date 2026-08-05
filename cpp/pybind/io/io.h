// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "pybind/open3d_pybind.h"

namespace open3d {
namespace io {

void pybind_io_declarations(py::module& m);
void pybind_class_io_declarations(py::module& m);
void pybind_rpc_declarations(py::module& m);
#ifdef BUILD_AZURE_KINECT
void pybind_sensor_declarations(py::module& m);
#endif

void pybind_io_definitions(py::module& m);
void pybind_class_io_definitions(py::module& m);
void pybind_rpc_definitions(py::module& m);
#ifdef BUILD_AZURE_KINECT
void pybind_sensor_definitions(py::module& m);
#endif

}  // namespace io
}  // namespace open3d
