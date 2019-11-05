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

#include "Open3D/Container/Blob.h"
#include "Open3D/Container/Device.h"
#include "Open3D/Container/MemoryManager.h"
#include "TestUtility/UnitTest.h"

#include "Container/ContainerTest.h"

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

TEST_P(BlobPermuteDevices, IsPtrInBlob) {
    Device device = GetParam();

    Blob b(10, Device(device));

    const char *head = static_cast<const char *>(b.v_);
    EXPECT_TRUE(b.IsPtrInBlob(head));
    EXPECT_TRUE(b.IsPtrInBlob(head + 9));
    EXPECT_FALSE(b.IsPtrInBlob(head + 10));
}
