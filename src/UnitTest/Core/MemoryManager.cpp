// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#include "Open3D/Core/MemoryManager.h"
#include "Open3D/Core/Blob.h"
#include "Open3D/Core/Device.h"

#include "Core/CoreTest.h"
#include "TestUtility/UnitTest.h"

#include <vector>

using namespace std;
using namespace open3d;

class MemoryManagerPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(MemoryManager,
                         MemoryManagerPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

class MemoryManagerPermuteDevicePairs : public PermuteDevicePairs {};
INSTANTIATE_TEST_SUITE_P(
        MemoryManager,
        MemoryManagerPermuteDevicePairs,
        testing::ValuesIn(MemoryManagerPermuteDevicePairs::TestCases()));

TEST_P(MemoryManagerPermuteDevices, MallocFree) {
    Device device = GetParam();

    void* ptr = MemoryManager::Malloc(10, device);
    MemoryManager::Free(ptr, device);
}

TEST_P(MemoryManagerPermuteDevicePairs, Memcpy) {
    Device dst_device;
    Device src_device;
    std::tie(dst_device, src_device) = GetParam();

    char dst_vals[6] = "xxxxx";
    char src_vals[6] = "hello";
    size_t num_bytes = strlen(src_vals) + 1;

    void* dst_ptr = MemoryManager::Malloc(num_bytes, dst_device);
    void* src_ptr = MemoryManager::Malloc(num_bytes, src_device);
    MemoryManager::MemcpyFromHost(src_ptr, src_device, (void*)src_vals,
                                  num_bytes);

    MemoryManager::Memcpy(dst_ptr, dst_device, src_ptr, src_device, num_bytes);
    MemoryManager::MemcpyToHost((void*)dst_vals, dst_ptr, dst_device,
                                num_bytes);
    ASSERT_STREQ(dst_vals, src_vals);

    MemoryManager::Free(dst_ptr, dst_device);
    MemoryManager::Free(src_ptr, src_device);
}
