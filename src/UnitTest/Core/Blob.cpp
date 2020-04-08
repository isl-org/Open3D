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

#include "Open3D/Core/Blob.h"
#include "Open3D/Core/Device.h"
#include "Open3D/Core/MemoryManager.h"
#include "TestUtility/UnitTest.h"

#include "Core/CoreTest.h"

using namespace std;
using namespace open3d;

class BlobPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(Blob,
                         BlobPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

TEST_P(BlobPermuteDevices, BlobConstructor) {
    Device device = GetParam();

    Blob b(10, Device(device));
}

TEST_P(BlobPermuteDevices, BlobConstructorWithExternalMemory) {
    Device device = GetParam();

    void* data_ptr = MemoryManager::Malloc(8, device);
    bool deleter_called = false;

    auto deleter = [&device, &deleter_called, data_ptr](void* dummy) -> void {
        MemoryManager::Free(data_ptr, device);
        deleter_called = true;
    };

    {
        Blob b(device, data_ptr, deleter);
        EXPECT_EQ(b.GetDataPtr(), data_ptr);
        EXPECT_FALSE(deleter_called);
    }
    EXPECT_TRUE(deleter_called);
}
