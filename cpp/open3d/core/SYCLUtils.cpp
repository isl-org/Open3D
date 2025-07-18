// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Contains code from:
// https://github.com/intel/llvm/blob/sycl/sycl/tools/sycl-ls/sycl-ls.cpp
// ----------------------------------------------------------------------------
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// ----------------------------------------------------------------------------

#include "open3d/core/SYCLUtils.h"

#include <array>
#include <cstdlib>
#include <sstream>

#include "open3d/utility/Logging.h"

#ifdef BUILD_SYCL_MODULE
#include <sycl/sycl.hpp>

#include "open3d/core/SYCLContext.h"
#endif

namespace open3d {
namespace core {
namespace sy {

int SYCLDemo() {
#ifdef BUILD_SYCL_MODULE
    // Ref: https://intel.github.io/llvm-docs/GetStartedGuide.html
    // Creating buffer of 4 ints to be used inside the kernel code.
    sycl::buffer<int, 1> buffer(4);

    // Creating SYCL queue.
    sycl::queue q;

    // Size of index space for kernel.
    sycl::range<1> num_workloads{buffer.size()};

    // Submitting command group(work) to q.
    q.submit([&](sycl::handler &cgh) {
        // Getting write only access to the buffer on a device.
        auto accessor = buffer.get_access<sycl::access::mode::write>(cgh);
        // Execute kernel.
        cgh.parallel_for<class FillBuffer>(
                num_workloads, [=](sycl::id<1> WIid) {
                    // Fill buffer with indexes.
                    accessor[WIid] = (int)WIid.get(0);
                });
    });

    // Getting access to the buffer on the host.
    // Implicit barrier waiting for q to complete the work.
    const auto host_accessor = buffer.get_host_access();

    // Check the results.
    bool mismatch_found = false;
    for (size_t i = 0; i < buffer.size(); ++i) {
        if (host_accessor[i] != i) {
            utility::LogInfo("Mismatch found at index {}: expected {}, got {}.",
                             i, i, host_accessor[i]);
            mismatch_found = true;
        }
    }

    if (mismatch_found) {
        utility::LogInfo("SYCLDemo failed!");
        return -1;
    } else {
        utility::LogInfo("SYCLDemo passed!");
        return 0;
    }
#else
    utility::LogInfo("SYCLDemo is not compiled with BUILD_SYCL_MODULE=ON.");
    return -1;
#endif
}

#ifdef BUILD_SYCL_MODULE

OPEN3D_DLL_LOCAL std::string GetDeviceTypeName(const sycl::device &device) {
    auto device_type = device.get_info<sycl::info::device::device_type>();
    switch (device_type) {
        case sycl::info::device_type::cpu:
            return "cpu";
        case sycl::info::device_type::gpu:
            return "gpu";
        case sycl::info::device_type::host:
            return "host";
        case sycl::info::device_type::accelerator:
            return "acc";
        case sycl::info::device_type::custom:
            return "custom";
        default:
            return "unknown";
    }
}

static std::string GetBackendName(const sycl::device &device) {
    sycl::platform platform = device.get_info<sycl::info::device::platform>();
    sycl::backend backend = platform.get_backend();
    std::ostringstream os;
    os << backend;
    return os.str();
}

static std::string SYCLDeviceToString(const sycl::device &device) {
    std::ostringstream os;
    os << "[" << GetBackendName(device) << ":" << GetDeviceTypeName(device)
       << "] " << device.get_info<sycl::info::device::name>();
    return os.str();
}
#endif

void PrintSYCLDevices(bool print_all) {
#ifdef BUILD_SYCL_MODULE
    const char *filter = std::getenv("ONEAPI_DEVICE_SELECTOR");
    if (filter) {
        utility::LogWarning(
                "ONEAPI_DEVICE_SELECTOR environment variable is set to {}. To "
                "see the correct device id, please unset "
                "ONEAPI_DEVICE_SELECTOR.",
                filter);
    }

    int nd = 0;
    utility::LogInfo("# Open3D SYCL device");
    try {
        auto dev = sycl::device(sycl::gpu_selector_v);
        utility::LogInfo("- Device(\"SYCL:{}\"): {}", nd,
                         SYCLDeviceToString(dev));
        ++nd;
    } catch (const sycl::exception &e) {  // No SYCL GPU available.
    }
    try {
        auto dev = sycl::device(sycl::cpu_selector_v);
        utility::LogInfo("# Open3D SYCL device (CPU fallback)");
        utility::LogInfo("- Device(\"SYCL:{}\"): {}", nd,
                         SYCLDeviceToString(dev));
    } catch (const sycl::exception &e) {  // No SYCL CPU available.
        if (nd == 0) utility::LogInfo("- Device(\"SYCL:0\"): N/A");
    }
    if (print_all) {
        utility::LogInfo("# All SYCL devices");
        const std::vector<sycl::platform> &platforms =
                sycl::platform::get_platforms();
        for (const sycl::platform &platform : platforms) {
            sycl::backend backend = platform.get_backend();
            const std::vector<sycl::device> &devices = platform.get_devices();
            for (const sycl::device &device : devices) {
                utility::LogInfo("- {}", SYCLDeviceToString(device));
            }
        }

        utility::LogInfo("# Default SYCL selectors");
        try {
            const sycl::device &device = sycl::device(sycl::default_selector_v);
            utility::LogInfo("- sycl::default_selector_v    : {}",
                             SYCLDeviceToString(device));
        } catch (const sycl::exception &e) {
            utility::LogInfo("- sycl::default_selector_v    : N/A");
        }
        try {
            const sycl::device &device = sycl::device(sycl::cpu_selector_v);
            utility::LogInfo("- sycl::cpu_selector_v        : {}",
                             SYCLDeviceToString(device));
        } catch (const sycl::exception &e) {
            utility::LogInfo("- sycl::cpu_selector_v        : N/A");
        }
        try {
            const sycl::device &device = sycl::device(sycl::gpu_selector_v);
            utility::LogInfo("- sycl::gpu_selector_v        : {}",
                             SYCLDeviceToString(device));
        } catch (const sycl::exception &e) {
            utility::LogInfo("- sycl::gpu_selector_v        : N/A");
        }
        try {
            const sycl::device &device =
                    sycl::device(sycl::accelerator_selector_v);
            utility::LogInfo("- sycl::accelerator_selector_v: {}",
                             SYCLDeviceToString(device));
        } catch (const sycl::exception &e) {
            utility::LogInfo("- sycl::accelerator_selector_v: N/A");
        }
    }

#else
    utility::LogInfo(
            "PrintSYCLDevices is not compiled with BUILD_SYCL_MODULE=ON.");
#endif
}

bool IsAvailable() {
#ifdef BUILD_SYCL_MODULE
    return SYCLContext::GetInstance().IsAvailable();
#else
    return false;
#endif
}

bool IsDeviceAvailable(const Device &device) {
#ifdef BUILD_SYCL_MODULE
    return SYCLContext::GetInstance().IsDeviceAvailable(device);
#else
    return false;
#endif
}

std::string GetDeviceType(const Device &device) {
#ifdef BUILD_SYCL_MODULE
    if (IsDeviceAvailable(device)) {
        return SYCLContext::GetInstance()
                .GetDeviceProperties(device)
                .device_type;
    } else {
        return "";
    }
#else
    return "";
#endif
}

std::vector<Device> GetAvailableSYCLDevices() {
#ifdef BUILD_SYCL_MODULE
    return SYCLContext::GetInstance().GetAvailableSYCLDevices();
#else
    return {};
#endif
}

void enablePersistentJITCache() {
#ifdef BUILD_SYCL_MODULE
#if defined(_WIN32)
    _putenv_s("SYCL_CACHE_PERSISTENT", "1");
#else
    setenv("SYCL_CACHE_PERSISTENT", "1", 1);
#endif
#else
    utility::LogInfo(
            "enablePersistentJITCache is not compiled with "
            "BUILD_SYCL_MODULE=ON.");
#endif
}

}  // namespace sy
}  // namespace core
}  // namespace open3d
