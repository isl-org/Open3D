// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/SYCLUtils.h"
#include "open3d/utility/Optional.h"
#include "pybind/core/core.h"

namespace open3d {
namespace core {

void pybind_sycl_utils_definitions(py::module& m) {
    m.def("sycl_demo", &sy::SYCLDemo);

    py::module m_sycl = m.def_submodule("sycl");
    m_sycl.def("is_available", sy::IsAvailable,
               "Returns true if Open3D is compiled with SYCL support and at "
               "least one compatible SYCL device is detected.");

    m_sycl.def("get_available_devices", sy::GetAvailableSYCLDevices,
               "Return a list of available SYCL devices.");

    m_sycl.def("print_sycl_devices", sy::PrintSYCLDevices,
               "print_all"_a = false,
               "Print SYCL device available to Open3D (either the best "
               "available GPU, or a fallback CPU device).  If `print_all` is "
               "specified, also print SYCL devices of other types.");

    m_sycl.def("enable_persistent_jit_cache", sy::enablePersistentJITCache,
               "Enables the JIT cache for SYCL. This sets an environment "
               "variable and "
               "will affect the entire process and any child processes.");

    m_sycl.def("get_device_type", sy::GetDeviceType, "device"_a,
               "Returns the device type (cpu / gpu / accelerator / custom) of "
               "the specified device as a string. Returns empty string if the "
               "device is not available.");
}

}  // namespace core
}  // namespace open3d
