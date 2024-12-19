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

#include <sycl/sycl.hpp>
#include <unordered_map>

#include "open3d/core/Device.h"

namespace open3d {
namespace core {
namespace sy {

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

private:
    SYCLContext();

    /// List of available Open3D SYCL devices.
    std::vector<Device> devices_;

    /// Maps core::Device to the corresponding default SYCL queue.
    std::unordered_map<Device, sycl::queue> device_to_default_queue_;

    /// Maps core::Device to sycl::device. Internal use only for now.
    std::unordered_map<Device, sycl::device> device_to_sycl_device_;
};

}  // namespace sy
}  // namespace core
}  // namespace open3d
