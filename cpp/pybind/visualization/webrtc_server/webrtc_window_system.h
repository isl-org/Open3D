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
namespace webrtc_server {

void pybind_webrtc_server_declarations(py::module &m);
void pybind_webrtc_server_definitions(py::module &m);

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
