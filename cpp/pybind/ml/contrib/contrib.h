// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "pybind/open3d_pybind.h"
#include "pybind/pybind_utils.h"

namespace open3d {
namespace ml {
namespace contrib {

void pybind_contrib_declarations(py::module &m);

void pybind_contrib_definitions(py::module &m);
void pybind_contrib_subsample_definitions(py::module &m_contrib);
void pybind_contrib_iou_definitions(py::module &m_contrib);

}  // namespace contrib
}  // namespace ml
}  // namespace open3d
