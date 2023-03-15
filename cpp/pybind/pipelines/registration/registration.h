// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "pybind/open3d_pybind.h"

namespace open3d {
namespace pipelines {
namespace registration {

void pybind_registration(py::module &m);

void pybind_feature(py::module &m);
void pybind_feature_methods(py::module &m);
void pybind_global_optimization(py::module &m);
void pybind_global_optimization_methods(py::module &m);
void pybind_robust_kernels(py::module &m);

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
