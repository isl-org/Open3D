// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/ml/contrib/contrib.h"

#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"
#include "pybind/pybind_utils.h"

namespace open3d {
namespace ml {
namespace contrib {

void pybind_contrib_declarations(py::module& m) {
    py::module m_contrib = m.def_submodule("contrib");
}
void pybind_contrib_definitions(py::module& m) {
    auto m_contrib = static_cast<py::module>(m.attr("contrib"));
    pybind_contrib_subsample_definitions(m_contrib);
    pybind_contrib_iou_definitions(m_contrib);
}

}  // namespace contrib
}  // namespace ml
}  // namespace open3d
