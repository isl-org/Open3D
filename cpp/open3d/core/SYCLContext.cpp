// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/SYCLContext.h"

#include <array>
#include <cstdlib>
#include <sstream>
#include <sycl/sycl.hpp>

#include "open3d/core/SYCLUtils.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace sy {

OPEN3D_DLL_LOCAL std::string GetDeviceTypeName(const sycl::device &device);

SYCLContext &SYCLContext::GetInstance() {
    static thread_local SYCLContext instance;
    return instance;
}

bool SYCLContext::IsAvailable() { return devices_.size() > 0; }

bool SYCLContext::IsDeviceAvailable(const Device &device) {
    return devices_.find(device) != devices_.end();
}
std::vector<Device> SYCLContext::GetAvailableSYCLDevices() {
    std::vector<Device> device_vec;
    for (const auto &device : devices_) {
        device_vec.push_back(device.first);
    }
    return device_vec;
}

sycl::queue SYCLContext::GetDefaultQueue(const Device &device) {
    return devices_.at(device).queue;
}

SYCLDevice::SYCLDevice(const sycl::device &sycl_device) {
    namespace sid = sycl::info::device;
    device = sycl_device;
    queue = sycl::queue(device);
    name = device.get_info<sid::name>();
    device_type = GetDeviceTypeName(device);
    max_work_group_size = device.get_info<sid::max_work_group_size>();
    auto aspects = device.get_info<sid::aspects>();
    fp64 = std::find(aspects.begin(), aspects.end(), sycl::aspect::fp64) !=
           aspects.end();
    if (!fp64) {
        utility::LogWarning(
                "SYCL device {} does not support double precision. Use env "
                "vars 'OverrideDefaultFP64Settings=1' "
                "'IGC_EnableDPEmulation=1' to enable double precision "
                "emulation on Intel GPUs.",
                name);
    }
    usm_device_allocations =
            std::find(aspects.begin(), aspects.end(),
                      sycl::aspect::usm_device_allocations) != aspects.end();
    if (!usm_device_allocations) {
        utility::LogWarning(
                "SYCL device {} does not support USM device allocations. "
                "Open3D SYCL support may not work.",
                name);
    }
}

SYCLContext::SYCLContext() {
    // SYCL GPU.
    // TODO: Currently we only support one GPU device.
    try {
        const sycl::device &sycl_device = sycl::device(sycl::gpu_selector_v);
        const Device open3d_device = Device("SYCL:0");
        devices_.emplace(open3d_device, sycl_device);
    } catch (const sycl::exception &e) {
        utility::LogWarning("SYCL GPU unavailable: {}", e.what());
    }

    // SYCL CPU fallback.
    // This could happen if the Intel GPGPU driver is not installed or if
    // your CPU does not have integrated GPU.
    try {
        if (devices_.size() == 0) {
            utility::LogWarning(
                    "SYCL GPU device is not available, falling back to SYCL "
                    "host device.");
        }
        const sycl::device &sycl_device = sycl::device(sycl::cpu_selector_v);
        const Device open3d_device =
                Device("SYCL:" + std::to_string(devices_.size()));
        devices_.emplace(open3d_device, sycl_device);
    } catch (const sycl::exception &e) {
        utility::LogWarning("SYCL CPU unavailable: {}", e.what());
    }

    if (devices_.size() == 0) {
        utility::LogWarning("No SYCL device is available.");
    }
}

}  // namespace sy
}  // namespace core
}  // namespace open3d
