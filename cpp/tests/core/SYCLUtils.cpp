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

#include "open3d/core/SYCLUtils.h"

#include <vector>

#include "open3d/core/MemoryManager.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/Timer.h"
#include "tests/Tests.h"
#include "tests/core/CoreTest.h"

namespace open3d {
namespace tests {

TEST(SYCLUtils, SYCLDemo) { core::sycl_utils::SYCLDemo(); }

TEST(SYCLUtils, PrintAllSYCLDevices) {
    core::sycl_utils::PrintSYCLDevices(/*print_all=*/true);
}

TEST(SYCLUtils, PrintSYCLDevices) {
    core::sycl_utils::PrintSYCLDevices(/*print_all=*/false);
}

TEST(SYCLUtils, SYCLMemoryModel) {
    if (core::sycl_utils::GetAvailableSYCLCPUDevices().empty() ||
        core::sycl_utils::GetAvailableSYCLGPUDevices().empty()) {
        // This test is to demonstrate SYCL memory model. Skip if no SYCL_GPU
        // is available.
        return;
    }

    size_t byte_size = sizeof(int) * 4;
    int* host_ptr = (int*)malloc(byte_size);
    for (int i = 0; i < 4; i++) {
        host_ptr[i] = i;
    }
    int* host_dst_ptr = (int*)malloc(byte_size);
    auto set_zero_host_dst_ptr = [&]() {
        for (int i = 0; i < 4; i++) {
            host_dst_ptr[i] = 0;
        }
    };
    set_zero_host_dst_ptr();
    core::Device host_device;

#ifdef ENABLE_SYCL_UNIFIED_SHARED_MEMORY
    utility::LogInfo("SYCLMemoryModel: unified shared memory");

    // Can host access SYCL_CPU's memory directly? Yes.
    core::Device sycl_cpu_device("SYCL_CPU:0");
    int* sycl_cpu_ptr =
            (int*)core::MemoryManager::Malloc(byte_size, sycl_cpu_device);
    core::MemoryManager::Memcpy(sycl_cpu_ptr, sycl_cpu_device, host_ptr,
                                host_device, byte_size);
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(sycl_cpu_ptr[i], i);
    }

    // Can host access SYCL_GPU's memory directly? Yes.
    core::Device sycl_gpu_device("SYCL_GPU:0");
    int* sycl_gpu_ptr =
            (int*)core::MemoryManager::Malloc(byte_size, sycl_gpu_device);
    core::MemoryManager::Memcpy(sycl_gpu_ptr, sycl_gpu_device, host_ptr,
                                host_device, byte_size);
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(sycl_gpu_ptr[i], i);
    }

#else
    utility::LogInfo("SYCLMemoryModel: device memory");

    // Can host access SYCL_CPU's memory directly? Yes.
    core::Device sycl_cpu_device("SYCL_CPU:0");
    int* sycl_cpu_ptr =
            (int*)core::MemoryManager::Malloc(byte_size, sycl_cpu_device);
    core::MemoryManager::Memcpy(sycl_cpu_ptr, sycl_cpu_device, host_ptr,
                                host_device, byte_size);
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(sycl_cpu_ptr[i], i);
    }

    // Can host access SYCL_GPU's memory directly? No.
    core::Device sycl_gpu_device("SYCL_GPU:0");
    int* sycl_gpu_ptr =
            (int*)core::MemoryManager::Malloc(byte_size, sycl_gpu_device);
    core::MemoryManager::Memcpy(sycl_gpu_ptr, sycl_gpu_device, host_ptr,
                                host_device, byte_size);
    for (int i = 0; i < 4; i++) {
        // This will segfault.
        // EXPECT_EQ(sycl_gpu_ptr[i], i);
    }
#endif

    free(host_ptr);
    free(host_dst_ptr);
    core::MemoryManager::Free(sycl_cpu_ptr, sycl_cpu_device);
    core::MemoryManager::Free(sycl_gpu_ptr, sycl_gpu_device);
}

}  // namespace tests
}  // namespace open3d
