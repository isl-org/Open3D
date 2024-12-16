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
namespace pipelines {
namespace odometry {

void pybind_odometry_declarations(py::module &m);
void pybind_odometry_definitions(py::module &m);

}  // namespace odometry
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
