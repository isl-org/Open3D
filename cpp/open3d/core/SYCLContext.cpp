// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/SYCLContext.h"

#include <CL/sycl.hpp>
#include <array>
#include <cstdlib>
#include <sstream>

#include "open3d/core/SYCLUtils.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace sycl {

SYCLContext &SYCLContext::GetInstance() {
    static thread_local SYCLContext instance;
    return instance;
}

bool SYCLContext::IsAvailable() { return devices_.size() > 0; }

bool SYCLContext::IsDeviceAvailable(const Device &device) {
    bool rc = false;
    for (const Device &device_ : devices_) {
        if (device == device_) {
            rc = true;
            break;
        }
    }
    return rc;
}
std::vector<Device> SYCLContext::GetAvailableSYCLDevices() { return devices_; }

sy::queue SYCLContext::GetDefaultQueue(const Device &device) {
    return device_to_default_queue_.at(device);
}

SYCLContext::SYCLContext() {
    // SYCL GPU.
    // TODO: Currently we only support one GPU device.
    try {
        const sy::device &sycl_device = sy::device(sy::gpu_selector());
        const Device open3d_device = Device("SYCL:0");
        devices_.push_back(open3d_device);
        device_to_sycl_device_[open3d_device] = sycl_device;
        device_to_default_queue_[open3d_device] = sy::queue(sycl_device);
    } catch (const sy::exception &e) {
    }

    if (devices_.size() == 0) {
        // SYCL CPU fallback.
        // This could happen if the Intel GPGPU driver is not installed or if
        // your CPU does not have integrated GPU.
        try {
            const sy::device &sycl_device = sy::device(sy::host_selector());
            const Device open3d_device = Device("SYCL:0");
            utility::LogWarning(
                    "SYCL GPU device is not available, falling back to SYCL "
                    "host device.");
            devices_.push_back(open3d_device);
            device_to_sycl_device_[open3d_device] = sycl_device;
            device_to_default_queue_[open3d_device] = sy::queue(sycl_device);
        } catch (const sy::exception &e) {
        }
    }

    if (devices_.size() == 0) {
        utility::LogWarning("No SYCL device is available.");
    }
}

}  // namespace sycl
}  // namespace core
}  // namespace open3d
