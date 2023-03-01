// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/t/t.h"

#include "pybind/open3d_pybind.h"
#include "pybind/t/geometry/geometry.h"
#include "pybind/t/io/io.h"
#include "pybind/t/pipelines/pipelines.h"

namespace open3d {
namespace t {

void pybind_t(py::module& m) {
    py::module m_submodule = m.def_submodule("t");
    pipelines::pybind_pipelines(m_submodule);
    geometry::pybind_geometry(m_submodule);
    io::pybind_io(m_submodule);
}

}  // namespace t
}  // namespace open3d
