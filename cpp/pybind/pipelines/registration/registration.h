// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "pybind/open3d_pybind.h"

namespace open3d {
namespace pipelines {
namespace registration {

void pybind_registration_declarations(py::module &m);
void pybind_feature_declarations(py::module &m_registration);
void pybind_global_optimization_declarations(py::module &m_registration);
void pybind_robust_kernels_declarations(py::module &m_registration);

void pybind_registration_definitions(py::module &m);
void pybind_feature_definitions(py::module &m_registration);
void pybind_global_optimization_definitions(py::module &m_registration);
void pybind_robust_kernels_definitions(py::module &m_registration);

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
