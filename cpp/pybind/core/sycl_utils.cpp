// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <optional>

#include "open3d/core/SYCLUtils.h"
#include "pybind/core/core.h"

namespace open3d {
namespace core {

void pybind_sycl_utils_definitions(py::module& m) {
    m.def("sycl_demo", &sy::SYCLDemo);

    py::module m_sycl = m.def_submodule("sycl");
    m_sycl.def("is_available", sy::IsAvailable,
               "Returns true if Open3D is compiled with SYCL support and at "
               "least one compatible SYCL device is detected.");

    m_sycl.def("device_count", sy::GetDeviceCount,
               "Return the number of available SYCL devices (including the CPU "
               "device).");

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

    py::class_<sy::SYCLDevice>(m_sycl, "SYCLDevice",
                               "Cached SYCL device properties.")
            .def_readonly("name", &sy::SYCLDevice::name)
            .def_readonly("device_type", &sy::SYCLDevice::device_type)
            .def_readonly("max_work_group_size",
                          &sy::SYCLDevice::max_work_group_size)
            .def_readonly("fp64", &sy::SYCLDevice::fp64)
            .def_readonly("usm_device_allocations",
                          &sy::SYCLDevice::usm_device_allocations)
            .def_readonly("discrete_gpu", &sy::SYCLDevice::discrete_gpu)
            .def_readonly("global_mem_size", &sy::SYCLDevice::global_mem_size);

    m_sycl.def("get_device_properties", &sy::GetSYCLDeviceProperties,
               "device"_a,
               "Return cached SYCL device properties, or a default-initialized "
               "SYCLDevice if the device is unavailable.");
}

}  // namespace core
}  // namespace open3d
