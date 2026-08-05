// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "pybind/open3d_pybind.h"

namespace open3d {
namespace pipelines {
namespace color_map {

void pybind_color_map_declarations(py::module &m);
void pybind_color_map_definitions(py::module &m);

}  // namespace color_map
}  // namespace pipelines
}  // namespace open3d
