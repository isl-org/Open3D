// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/// \file SYCLContext.h
/// \brief SYCL device properties and (when built) queue manager.
///
/// \ref SYCLDevice is always defined (host-visible POD). \ref SYCLContext and
/// SYCL runtime handles live behind BUILD_SYCL_MODULE=ON in SYCLContext.cpp.
///
/// Generic host TUs may include this header for \ref SYCLDevice only; SYCL
/// kernel TUs include the full SYCL API via SYCL_LANGUAGE_VERSION.

#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "open3d/core/Device.h"

#ifdef BUILD_SYCL_MODULE
// Host TUs forward-declare sycl::queue; SYCL-compiled TUs include the runtime.
#if defined(SYCL_LANGUAGE_VERSION)
#include <sycl/sycl.hpp>
#else
namespace sycl {
class queue;
}
#endif
#endif

namespace open3d {
namespace core {
namespace sy {

/// Cached SYCL device properties (no SYCL runtime types in this struct).
struct SYCLDevice {
    std::string name{};         ///< Friendly / descriptive name of the device.
    std::string device_type{};  ///< cpu, gpu, host, acc, custom, unknown.
    size_t max_work_group_size = 0;  ///< Device max work-group size.
    bool fp64 = false;  ///< Native fp64; else may need emulation on some GPUs.
    bool usm_device_allocations =
            false;  ///< USM device allocations required for Open3D SYCL.
    /// Discrete GPU: device_type is gpu and not host-unified (integrated)
    /// memory.
    bool discrete_gpu = false;
    uint64_t global_mem_size = 0;  ///< Global memory size in bytes.
    /// Sub-group (SIMD/wave) widths natively supported by the device, e.g.
    /// {8, 16, 32} on discrete Arc GPUs vs {16, 32} on some integrated Xe
    /// GPUs.
    std::vector<size_t> sub_group_sizes{};

    /// True if \p size is one of the device's natively supported sub-group
    /// widths (safe to request via `[[sycl::reqd_sub_group_size(size)]]`).
    bool SupportsSubgroupSize(size_t size) const {
        return std::find(sub_group_sizes.begin(), sub_group_sizes.end(),
                         size) != sub_group_sizes.end();
    }
};

#ifdef BUILD_SYCL_MODULE

/// Singleton SYCL context manager. It maintains:
/// - A default in-order queue per Open3D SYCL device
/// - Cached \ref SYCLDevice properties for each device
class SYCLContext {
public:
    SYCLContext(SYCLContext const&) = delete;
    void operator=(SYCLContext const&) = delete;
    ~SYCLContext();

    /// Get singleton instance (process-wide).
    static SYCLContext& GetInstance();

    /// Returns true if there is at least one SYCL device.
    bool IsAvailable();

    /// Returns true if the specified SYCL device is available.
    bool IsDeviceAvailable(const Device& device);

    /// Returns a list of all available SYCL devices.
    std::vector<Device> GetAvailableSYCLDevices();

    /// Get the default SYCL queue given an Open3D device.
    sycl::queue GetDefaultQueue(const Device& device);

    /// Get cached SYCL device properties, or a default-initialized \ref
    /// SYCLDevice if \p device is not available.
    SYCLDevice GetDeviceProperties(const Device& device);

private:
    SYCLContext();

    /// Holds sycl::device / sycl::queue (defined only in SYCLContext.cpp).
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif  // BUILD_SYCL_MODULE

}  // namespace sy
}  // namespace core
}  // namespace open3d
