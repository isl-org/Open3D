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
