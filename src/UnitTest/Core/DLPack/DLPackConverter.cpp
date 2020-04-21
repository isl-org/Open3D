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

#include "Open3D/Core/DLPack/DLPackConverter.h"
#include "Open3D/Core/DLPack/dlpack.h"
#include "Open3D/Core/Tensor.h"

#include "Core/CoreTest.h"
#include "TestUtility/UnitTest.h"

#include <vector>

using namespace std;
using namespace open3d;

class DLPackPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(Tensor,
                         DLPackPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

class DLPackPermuteDevicePairs : public PermuteDevicePairs {};
INSTANTIATE_TEST_SUITE_P(
        Tensor,
        DLPackPermuteDevicePairs,
        testing::ValuesIn(DLPackPermuteDevicePairs::TestCases()));

TEST_P(DLPackPermuteDevices, ToDLPackFromDLPack) {
    Device device = GetParam();
    std::vector<float> vals{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};

    Tensor src_t(vals, {2, 3, 4}, Dtype::Float32, device);
    const void *blob_head = src_t.GetBlob()->GetDataPtr();

    // src_t = src_t[1, 0:3:2, 0:4:2], a mix of [] and slice
    src_t = src_t[1].Slice(0, 0, 3, 2).Slice(1, 0, 4, 2);
    EXPECT_EQ(src_t.GetShape(), SizeVector({2, 2}));
    EXPECT_EQ(src_t.GetStrides(), SizeVector({8, 2}));
    EXPECT_EQ(src_t.GetBlob()->GetDataPtr(), blob_head);
    EXPECT_EQ(src_t.GetDataPtr(),
              static_cast<const char *>(blob_head) +
                      DtypeUtil::ByteSize(Dtype::Float32) * 3 * 4);
    EXPECT_EQ(src_t.ToFlatVector<float>(),
              std::vector<float>({12, 14, 20, 22}));

    DLManagedTensor *dl_t = src_t.ToDLPack();

    Tensor dst_t = Tensor::FromDLPack(dl_t);
    EXPECT_EQ(dst_t.GetShape(), SizeVector({2, 2}));
    EXPECT_EQ(dst_t.GetStrides(), SizeVector({8, 2}));
    // Note that the original blob head's info has been discarded.
    EXPECT_EQ(dst_t.GetBlob()->GetDataPtr(),
              static_cast<const char *>(blob_head) +
                      DtypeUtil::ByteSize(Dtype::Float32) * 3 * 4);
    EXPECT_EQ(dst_t.GetDataPtr(),
              static_cast<const char *>(blob_head) +
                      DtypeUtil::ByteSize(Dtype::Float32) * 3 * 4);
    EXPECT_EQ(dst_t.ToFlatVector<float>(),
              std::vector<float>({12, 14, 20, 22}));
}
