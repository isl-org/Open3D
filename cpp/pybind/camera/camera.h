// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "pybind/open3d_pybind.h"

namespace open3d {
namespace camera {

void pybind_camera_declarations(py::module &m);
void pybind_camera_definitions(py::module &m);

}  // namespace camera
}  // namespace open3d
