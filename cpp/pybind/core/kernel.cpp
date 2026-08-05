// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/kernel/Kernel.h"

#include "pybind/core/core.h"
#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"

namespace open3d {
namespace core {

void pybind_core_kernel_declarations(py::module &m) {
    py::module m_kernel = m.def_submodule("kernel");
}
void pybind_core_kernel_definitions(py::module &m) {
    auto m_kernel = static_cast<py::module>(m.attr("kernel"));
    m_kernel.def("test_linalg_integration",
                 &core::kernel::TestLinalgIntegration);
}

}  // namespace core
}  // namespace open3d
