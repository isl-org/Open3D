// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/t/io/io.h"

#include "pybind/open3d_pybind.h"

namespace open3d {
namespace t {
namespace io {

void pybind_io(py::module& m) {
    py::module m_io =
            m.def_submodule("io", "Tensor-based input-output handling module.");
    pybind_class_io(m_io);
    pybind_sensor(m_io);
}

}  // namespace io
}  // namespace t
}  // namespace open3d
