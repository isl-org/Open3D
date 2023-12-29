// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/Blob.h"

#include "open3d/core/Device.h"
#include "open3d/core/MemoryManager.h"
#include "tests/Tests.h"
#include "tests/core/CoreTest.h"

namespace open3d {
namespace tests {

class BlobPermuteDevices : public PermuteDevicesWithSYCL {};
INSTANTIATE_TEST_SUITE_P(
        Blob,
        BlobPermuteDevices,
        testing::ValuesIn(PermuteDevicesWithSYCL::TestCases()));

TEST_P(BlobPermuteDevices, BlobConstructor) {
    core::Device device = GetParam();

    core::Blob b(10, core::Device(device));
}

TEST_P(BlobPermuteDevices, BlobConstructorWithExternalMemory) {
    core::Device device = GetParam();

    void* data_ptr = core::MemoryManager::Malloc(8, device);
    bool deleter_called = false;

    auto deleter = [&device, &deleter_called, data_ptr](void* dummy) -> void {
        core::MemoryManager::Free(data_ptr, device);
        deleter_called = true;
    };

    {
        core::Blob b(device, data_ptr, deleter);
        EXPECT_EQ(b.GetDataPtr(), data_ptr);
        EXPECT_FALSE(deleter_called);
    }
    EXPECT_TRUE(deleter_called);
}

}  // namespace tests
}  // namespace open3d
