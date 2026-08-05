// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/core/Tensor.h"
#include "pybind/open3d_pybind.h"

namespace open3d {
namespace core {

void pybind_core_declarations(py::module& m);
void pybind_cuda_utils_declarations(py::module& m);
void pybind_core_blob_declarations(py::module& m);
void pybind_core_dtype_declarations(py::module& m);
void pybind_core_device_declarations(py::module& m);
void pybind_core_size_vector_declarations(py::module& m);
void pybind_core_tensor_declarations(py::module& m);
void pybind_core_tensor_accessor(py::class_<Tensor>& t);
void pybind_core_tensor_function_definitions(py::module& m);
void pybind_core_kernel_declarations(py::module& m);
void pybind_core_hashmap_declarations(py::module& m);
void pybind_core_hashset_declarations(py::module& m);
void pybind_core_scalar_declarations(py::module& m);

void pybind_core_definitions(py::module& m);
void pybind_cuda_utils_definitions(py::module& m);
void pybind_sycl_utils_definitions(py::module& m);
void pybind_core_dtype_definitions(py::module& m);
void pybind_core_device_definitions(py::module& m);
void pybind_core_size_vector_definitions(py::module& m);
void pybind_core_tensor_definitions(py::module& m);
void pybind_core_linalg_definitions(py::module& m);
void pybind_core_kernel_definitions(py::module& m);
void pybind_core_hashmap_definitions(py::module& m);
void pybind_core_hashset_definitions(py::module& m);
void pybind_core_scalar_definitions(py::module& m);

}  // namespace core
}  // namespace open3d
