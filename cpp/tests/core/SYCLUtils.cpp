// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
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

TEST(SYCLUtils, SYCLDemo) { core::sycl::SYCLDemo(); }

TEST(SYCLUtils, PrintAllSYCLDevices) {
    core::sycl::PrintSYCLDevices(/*print_all=*/true);
}

TEST(SYCLUtils, PrintSYCLDevices) {
    core::sycl::PrintSYCLDevices(/*print_all=*/false);
}

TEST(SYCLUtils, SYCLUnifiedSharedMemory) {
    if (!core::sycl::IsAvailable()) {
        return;
    }

    size_t byte_size = sizeof(int) * 4;
    int* host_ptr = static_cast<int*>(malloc(byte_size));
    for (int i = 0; i < 4; i++) {
        host_ptr[i] = i;
    }
    core::Device host_device;

#ifdef ENABLE_SYCL_UNIFIED_SHARED_MEMORY
    utility::LogInfo("SYCLMemoryModel: unified shared memory");
    // Can host access SYCL GPU's memory directly? Yes.
    core::Device sycl_device("SYCL:0");
    int* sycl_ptr = static_cast<int*>(
            core::MemoryManager::Malloc(byte_size, sycl_device));
    core::MemoryManager::Memcpy(sycl_ptr, sycl_device, host_ptr, host_device,
                                byte_size);
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(sycl_ptr[i], i);
    }
#else
    utility::LogInfo("SYCLMemoryModel: device memory");
    // Can host access SYCL GPU's memory directly? No.
    core::Device sycl_device("SYCL:0");
    int* sycl_ptr = static_cast<int*>(
            core::MemoryManager::Malloc(byte_size, sycl_device));
    core::MemoryManager::Memcpy(sycl_ptr, sycl_device, host_ptr, host_device,
                                byte_size);
    for (int i = 0; i < 4; i++) {
        // EXPECT_EQ(sycl_ptr[i], i); // This will segfault.
    }
#endif

    free(host_ptr);
    core::MemoryManager::Free(sycl_ptr, sycl_device);
}

}  // namespace tests
}  // namespace open3d
