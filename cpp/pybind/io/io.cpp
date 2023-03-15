// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/io/io.h"

#include "pybind/open3d_pybind.h"

namespace open3d {
namespace io {

void pybind_io(py::module &m) {
    py::module m_io = m.def_submodule("io");
    pybind_class_io(m_io);
    pybind_rpc(m_io);
#ifdef BUILD_AZURE_KINECT
    pybind_sensor(m_io);
#endif
}

}  // namespace io
}  // namespace open3d
