// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/// \file SYCLUtils.h
/// \brief Common SYCL utilities
///
/// SYCLUtils.h and SYCLUtils.cpp should compile when BUILD_SYCL_MODULE=ON or
/// BUILD_SYCL_MODULE=OFF. Use macros for conditional compilation.

#pragma once

#include <vector>

#include "open3d/core/Device.h"

namespace open3d {
namespace core {
namespace sy {

/// Runs simple SYCL test program for sanity checks.
/// \return Retuns 0 if successful.
int SYCLDemo();

/// Print available SYCL devices.
///
/// \param print_all If true, prints all SYCL devices. Otherwise, prints only
/// devices that are available for Open3D.
void PrintSYCLDevices(bool print_all = false);

/// Returns true if there is at least one SYCL device available.
bool IsAvailable();

/// Returns true if the specified SYCL device is available.
bool IsDeviceAvailable(const Device& device);

/// Returns the device type (cpu / gpu / accelerator / custom) of the specified
/// device as a string. Returns empty string if the device is not available.
std::string GetDeviceType(const Device& device);

/// Return a list of available SYCL devices.
std::vector<Device> GetAvailableSYCLDevices();

/// Enables the JIT cache for SYCL. This sets an environment variable and will
/// affect the entire process and any child processes.
void enablePersistentJITCache();

}  // namespace sy
}  // namespace core
}  // namespace open3d
