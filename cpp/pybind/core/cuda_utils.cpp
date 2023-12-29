// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/CUDAUtils.h"
#include "open3d/utility/Optional.h"
#include "pybind/core/core.h"

namespace open3d {
namespace core {

void pybind_cuda_utils(py::module& m) {
    py::module m_cuda = m.def_submodule("cuda");

    m_cuda.def("device_count", cuda::DeviceCount,
               "Returns the number of available CUDA devices. Returns 0 if "
               "Open3D is not compiled with CUDA support.");
    m_cuda.def("is_available", cuda::IsAvailable,
               "Returns true if Open3D is compiled with CUDA support and at "
               "least one compatible CUDA device is detected.");
    m_cuda.def("release_cache", cuda::ReleaseCache,
               "Releases CUDA memory manager cache. This is typically used for "
               "debugging.");
    m_cuda.def(
            "synchronize",
            [](const utility::optional<Device>& device) {
                if (device.has_value()) {
                    cuda::Synchronize(device.value());
                } else {
                    cuda::Synchronize();
                }
            },
            "Synchronizes CUDA devices. If no device is specified, all CUDA "
            "devices will be synchronized. No effect if the specified device "
            "is not a CUDA device. No effect if Open3D is not compiled with "
            "CUDA support.",
            "device"_a = py::none());
}

}  // namespace core
}  // namespace open3d
