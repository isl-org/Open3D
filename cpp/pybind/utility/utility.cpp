// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/utility/utility.h"

#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"

namespace open3d {
namespace utility {

void pybind_utility(py::module &m) {
    py::module m_submodule = m.def_submodule("utility");
    pybind_eigen(m_submodule);
    pybind_logging(m_submodule);
    random::pybind_random(m_submodule);
}

}  // namespace utility
}  // namespace open3d
