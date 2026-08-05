// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/core/core.h"

#include "open3d/core/Tensor.h"
#include "open3d/utility/Logging.h"
#include "pybind/core/nns/nearest_neighbor_search.h"
#include "pybind/open3d_pybind.h"
#include "pybind/pybind_utils.h"

namespace open3d {
namespace core {

void pybind_core_declarations(py::module& m) {
    py::module m_core = m.def_submodule("core");

    // opn3d::core namespace.
    pybind_cuda_utils_declarations(m_core);
    pybind_core_blob_declarations(m_core);
    pybind_core_dtype_declarations(m_core);
    pybind_core_device_declarations(m_core);
    pybind_core_size_vector_declarations(m_core);
    pybind_core_tensor_declarations(m_core);
    pybind_core_kernel_declarations(m_core);
    pybind_core_hashmap_declarations(m_core);
    pybind_core_hashset_declarations(m_core);
    pybind_core_scalar_declarations(m_core);

    // opn3d::core::nns namespace.
    py::module m_nns = m_core.def_submodule("nns");
    nns::pybind_core_nns_declarations(m_nns);
}

void pybind_core_definitions(py::module& m) {
    auto m_core = static_cast<py::module>(m.attr("core"));
    pybind_cuda_utils_definitions(m_core);
    pybind_sycl_utils_definitions(m_core);
    pybind_core_dtype_definitions(m_core);
    pybind_core_device_definitions(m_core);
    pybind_core_size_vector_definitions(m_core);
    pybind_core_tensor_definitions(m_core);
    pybind_core_tensor_function_definitions(m_core);
    pybind_core_linalg_definitions(m_core);
    pybind_core_kernel_definitions(m_core);
    pybind_core_hashmap_definitions(m_core);
    pybind_core_hashset_definitions(m_core);
    pybind_core_scalar_definitions(m_core);
    auto m_nns = static_cast<py::module>(m_core.attr("nns"));
    nns::pybind_core_nns_definitions(m_nns);
}

}  // namespace core
}  // namespace open3d
