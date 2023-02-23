// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/core/Tensor.h"
#include "pybind/open3d_pybind.h"

namespace open3d {
namespace core {

void pybind_core(py::module& m);
void pybind_cuda_utils(py::module& m);
void pybind_sycl_utils(py::module& m);
void pybind_core_blob(py::module& m);
void pybind_core_dtype(py::module& m);
void pybind_core_device(py::module& m);
void pybind_core_size_vector(py::module& m);
void pybind_core_tensor(py::module& m);
void pybind_core_tensor_accessor(py::class_<Tensor>& t);
void pybind_core_tensor_function(py::module& m);
void pybind_core_linalg(py::module& m);
void pybind_core_kernel(py::module& m);
void pybind_core_hashmap(py::module& m);
void pybind_core_hashset(py::module& m);
void pybind_core_scalar(py::module& m);

}  // namespace core
}  // namespace open3d
