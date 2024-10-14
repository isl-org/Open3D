// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/utility/utility.h"

#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"

namespace open3d {
namespace utility {

void pybind_utility_declarations(py::module &m) {
    py::module m_utility = m.def_submodule("utility");
    pybind_eigen_declarations(m_utility);
    pybind_logging_declarations(m_utility);
    random::pybind_random(m_utility);
}
void pybind_utility_definitions(py::module &m) {
    auto m_utility = static_cast<py::module>(m.attr("utility"));
    pybind_eigen_definitions(m_utility);
    pybind_logging_definitions(m_utility);
}

}  // namespace utility
}  // namespace open3d
