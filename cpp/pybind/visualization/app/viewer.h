// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "pybind/open3d_pybind.h"

namespace open3d {
namespace visualization {
namespace app {

void pybind_app_declarations(py::module &m);
void pybind_app_definitions(py::module &m);

}  // namespace app
}  // namespace visualization
}  // namespace open3d
