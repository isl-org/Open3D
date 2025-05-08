// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/// \file SYCLContext.h
/// \brief SYCL queue manager.
///
/// Unlike from SYCLUtils.h, SYCLContext.h shall only be included by source
/// files that are compiled with SYCL flags. Other generic source files (e.g.,
/// Device.cpp) shall not include this file.

#pragma once

#include <map>
#include <sycl/sycl.hpp>

#include "open3d/core/Device.h"

namespace open3d {
namespace core {
namespace sy {

/// @brief SYCL device properties.
struct SYCLDevice {
    SYCLDevice(const sycl::device& sycl_device);
    std::string name;            ///< Fiendlly / descriptive name of the device.
    std::string device_type;     ///< cpu, gpu, host, acc, custom, unknown.
    sycl::device device;         ///< SYCL device.
    sycl::queue queue;           ///< Default queue for this device.
    size_t max_work_group_size;  ///< Preferred work group size
    bool fp64;  ///< Double precision support, else need to emulate.
    bool usm_device_allocations;  ///< USM device allocations required for
                                  ///< Open3D.
};

/// Singleton SYCL context manager. It maintains:
/// - A default queue for each SYCL device
class SYCLContext {
public:
    SYCLContext(SYCLContext const&) = delete;
    void operator=(SYCLContext const&) = delete;

    /// Get singleton instance.
    static SYCLContext& GetInstance();

    /// Returns true if there is at least one SYCL devices.
    bool IsAvailable();

    /// Returns true if the specified SYCL device is available.
    bool IsDeviceAvailable(const Device& device);

    /// Returns a list of all available SYCL devices.
    std::vector<Device> GetAvailableSYCLDevices();

    /// Get the default SYCL queue given an Open3D device.
    sycl::queue GetDefaultQueue(const Device& device);

    /// Get SYCL device properties given an Open3D device.
    SYCLDevice GetDeviceProperties(const Device& device) {
        return devices_.at(device);
    };

private:
    SYCLContext();

    /// Map from available Open3D SYCL devices to their properties.
    std::map<Device, SYCLDevice> devices_;
};

}  // namespace sy
}  // namespace core
}  // namespace open3d
