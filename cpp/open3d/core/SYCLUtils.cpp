// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
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
#include <CL/sycl.hpp>

#include "open3d/core/SYCLContext.h"
#endif

namespace open3d {
namespace core {
namespace sycl {

#ifdef BUILD_SYCL_MODULE
namespace sy = cl::sycl;
#endif

int SYCLDemo() {
#ifdef BUILD_SYCL_MODULE
    // Ref: https://intel.github.io/llvm-docs/GetStartedGuide.html
    // Creating buffer of 4 ints to be used inside the kernel code.
    sy::buffer<sy::cl_int, 1> buffer(4);

    // Creating SYCL queue.
    sy::queue q;

    // Size of index space for kernel.
    sy::range<1> num_workloads{buffer.size()};

    // Submitting command group(work) to q.
    q.submit([&](sy::handler &cgh) {
        // Getting write only access to the buffer on a device.
        auto accessor = buffer.get_access<sy::access::mode::write>(cgh);
        // Execute kernel.
        cgh.parallel_for<class FillBuffer>(num_workloads, [=](sy::id<1> WIid) {
            // Fill buffer with indexes.
            accessor[WIid] = (sy::cl_int)WIid.get(0);
        });
    });

    // Getting read only access to the buffer on the host.
    // Implicit barrier waiting for q to complete the work.
    const auto host_accessor = buffer.get_access<sy::access::mode::read>();

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

static std::string GetDeviceTypeName(const sy::device &device) {
    auto device_type = device.get_info<sy::info::device::device_type>();
    switch (device_type) {
        case sy::info::device_type::cpu:
            return "cpu";
        case sy::info::device_type::gpu:
            return "gpu";
        case sy::info::device_type::host:
            return "host";
        case sy::info::device_type::accelerator:
            return "acc";
        default:
            return "unknown";
    }
}

static std::string GetBackendName(const sy::device &device) {
    sy::platform platform = device.get_info<sy::info::device::platform>();
    sy::backend backend = platform.get_backend();
    std::ostringstream os;
    os << backend;
    return os.str();
}

static std::string SYCLDeviceToString(const sy::device &device) {
    std::ostringstream os;
    os << "[" << GetBackendName(device) << ":" << GetDeviceTypeName(device)
       << "] " << device.get_info<sy::info::device::name>();
    return os.str();
}
#endif

void PrintSYCLDevices(bool print_all) {
#ifdef BUILD_SYCL_MODULE
    const char *filter = std::getenv("SYCL_DEVICE_FILTER");
    if (filter) {
        utility::LogWarning(
                "SYCL_DEVICE_FILTER environment variable is set to {}. To see "
                "the correct device id, please unset SYCL_DEVICE_FILTER.",
                filter);
    }

    if (print_all) {
        utility::LogInfo("# All SYCL devices");
        const std::vector<sy::platform> &platforms =
                sy::platform::get_platforms();
        for (const sy::platform &platform : platforms) {
            sy::backend backend = platform.get_backend();
            const std::vector<sy::device> &devices = platform.get_devices();
            for (const sy::device &device : devices) {
                utility::LogInfo("- {}", SYCLDeviceToString(device));
            }
        }

        utility::LogInfo("# Default SYCL selectors");
        try {
            const sy::device &device = sy::device(sy::default_selector());
            utility::LogInfo("- sycl::default_selector()    : {}",
                             SYCLDeviceToString(device));
        } catch (const sy::exception &e) {
            utility::LogInfo("- sycl::default_selector()    : N/A");
        }
        try {
            const sy::device &device = sy::device(sy::host_selector());
            utility::LogInfo("- sycl::host_selector()       : {}",
                             SYCLDeviceToString(device));
        } catch (const sy::exception &e) {
            utility::LogInfo("- sycl::host_selector()       : N/A");
        }
        try {
            const sy::device &device = sy::device(sy::cpu_selector());
            utility::LogInfo("- sycl::cpu_selector()        : {}",
                             SYCLDeviceToString(device));
        } catch (const sy::exception &e) {
            utility::LogInfo("- sycl::cpu_selector()        : N/A");
        }
        try {
            const sy::device &device = sy::device(sy::gpu_selector());
            utility::LogInfo("- sycl::gpu_selector()        : {}",
                             SYCLDeviceToString(device));
        } catch (const sy::exception &e) {
            utility::LogInfo("- sycl::gpu_selector()        : N/A");
        }
        try {
            const sy::device &device = sy::device(sy::accelerator_selector());
            utility::LogInfo("- sycl::accelerator_selector(): {}",
                             SYCLDeviceToString(device));
        } catch (const sy::exception &e) {
            utility::LogInfo("- sycl::accelerator_selector(): N/A");
        }

        utility::LogInfo("# Open3D SYCL device");
        try {
            const sy::device &device = sy::device(sy::gpu_selector());
            utility::LogInfo("- Device(\"SYCL:0\"): {}",
                             SYCLDeviceToString(device));
        } catch (const sy::exception &e) {
            utility::LogInfo("- Device(\"SYCL:0\"): N/A");
        }
    } else {
        utility::LogInfo("# Open3D SYCL device");
        try {
            const sy::device &device = sy::device(sy::gpu_selector());
            utility::LogInfo("- Device(\"SYCL:0\"): {}",
                             SYCLDeviceToString(device));
        } catch (const sy::exception &e) {
            utility::LogInfo("- Device(\"SYCL:0\"): N/A");
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

std::vector<Device> GetAvailableSYCLDevices() {
#ifdef BUILD_SYCL_MODULE
    return SYCLContext::GetInstance().GetAvailableSYCLDevices();
#else
    return {};
#endif
}

}  // namespace sycl
}  // namespace core
}  // namespace open3d
