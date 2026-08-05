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
namespace integration {

void pybind_integration_declarations(py::module &m);
void pybind_integration_definitions(py::module &m);

}  // namespace integration
}  // namespace pipelines
}  // namespace open3d
