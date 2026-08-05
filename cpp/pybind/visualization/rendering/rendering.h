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
namespace rendering {

void pybind_rendering_declarations(py::module &m);
void pybind_rendering_definitions(py::module &m);

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
