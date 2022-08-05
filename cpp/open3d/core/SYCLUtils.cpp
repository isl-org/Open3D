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

#include "open3d/core/MemoryManager.h"
#include "open3d/utility/Logging.h"

#ifdef BUILD_SYCL_MODULE
#include <CL/sycl.hpp>

#include "open3d/core/Indexer.h"
#include "open3d/core/SYCLContext.h"
#include "open3d/core/Tensor.h"
#endif

namespace open3d {
namespace core {
namespace sycl {

#if defined(BUILD_SYCL_MODULE)
sy::queue &GetDefaultQueue(const Device &device) {
    return SYCLContext::GetInstance().GetDefaultQueue(device);
}
#endif

int SYCLDemo() {
#ifdef BUILD_SYCL_MODULE
    int n = 4;

    Device cpu_device("CPU:0");
    Device sycl_device("SYCL:0");
    core::Tensor src = core::Tensor::Init<float>({0, 1, 2, 3}, cpu_device);
    core::Tensor dst = src.To(sycl_device);
    utility::LogInfo("{}", dst.ToString());

    return 0;
#else
    utility::LogInfo("SYCLDemo is not compiled with BUILD_SYCL_MODULE=ON.");
    return -1;
#endif
}

int SYCLDemoOld() {
#ifdef BUILD_SYCL_MODULE
    int n = 4;
    int num_bytes = n * sizeof(int);

    // Malloc.
    Device host_device("CPU:0");
    Device sycl_device("SYCL:0");
    int *host_buffer =
            static_cast<int *>(MemoryManager::Malloc(num_bytes, host_device));
    int *sycl_buffer =
            static_cast<int *>(MemoryManager::Malloc(num_bytes, sycl_device));

    // Prepare host buffer.
    for (int i = 0; i < n; i++) {
        host_buffer[i] = i;
    }

    // Copy to device.
    MemoryManager::Memcpy(sycl_buffer, sycl_device, host_buffer, host_device,
                          num_bytes);

    // Compute, every element +10.
    sy::queue &queue = sycl::GetDefaultQueue(sycl_device);
    queue.submit([&](sy::handler &h) {
             h.parallel_for(n, [=](int i) { sycl_buffer[i] += 10; });
         }).wait();

    // Copy back to host.
    MemoryManager::Memcpy(host_buffer, host_device, sycl_buffer, sycl_device,
                          num_bytes);

    // Check results.
    bool all_match = true;
    for (int i = 0; i < n; i++) {
        if (host_buffer[i] != i + 10) {
            all_match = false;
            utility::LogInfo("Mismatch: host_buffer[{}] = {}, expected {}.", i,
                             host_buffer[i], i + 10);
        } else {
            utility::LogInfo("Match: host_buffer[{}] = {}.", i, host_buffer[i]);
        }
    }

    // Clean up.
    MemoryManager::Free(host_buffer, host_device);
    MemoryManager::Free(sycl_buffer, sycl_device);

    return all_match ? 0 : -1;
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
