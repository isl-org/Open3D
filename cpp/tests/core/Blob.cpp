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
