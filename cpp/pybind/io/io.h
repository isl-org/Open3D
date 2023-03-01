// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "pybind/open3d_pybind.h"

namespace open3d {
namespace io {

void pybind_io(py::module& m);

void pybind_class_io(py::module& m);

void pybind_rpc(py::module& m);

#ifdef BUILD_AZURE_KINECT
void pybind_sensor(py::module& m);
#endif

}  // namespace io
}  // namespace open3d
