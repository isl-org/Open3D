// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/ml/ml.h"

#include "pybind/ml/contrib/contrib.h"
#include "pybind/open3d_pybind.h"

namespace open3d {
namespace ml {

void pybind_ml_declarations(py::module &m) {
    py::module m_ml = m.def_submodule("ml");
    contrib::pybind_contrib_declarations(m_ml);
}

void pybind_ml_definitions(py::module &m) {
    auto m_ml = static_cast<py::module>(m.attr("ml"));
    contrib::pybind_contrib_definitions(m_ml);
}

}  // namespace ml
}  // namespace open3d
