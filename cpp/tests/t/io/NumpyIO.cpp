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

#include "open3d/t/io/NumpyIO.h"

#include <cmath>
#include <limits>

#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"
#include "tests/UnitTest.h"
#include "tests/core/CoreTest.h"

namespace open3d {
namespace tests {

class NumpyIOPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(Tensor,
                         NumpyIOPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

TEST_P(NumpyIOPermuteDevices, NpyIO) {
    const core::Device &device = GetParam();
    const std::string file_name = "tensor.npy";

    core::Tensor t;
    core::Tensor t_load;

    // 2x2 tensor.
    t = core::Tensor::Init<float>({{1, 2}, {3, 4}}, device);
    t.Save(file_name);
    t_load = core::Tensor::Load(file_name);
    EXPECT_TRUE(t.AllClose(t_load.To(device)));

    // Non-contiguous tensor will be stored as contiguous tensor.
    t = core::Tensor::Init<float>(
            {{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
             {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}},
            device);
    // t[0:2:1, 0:3:2, 0:4:2]
    t = t.Slice(0, 0, 2, 1).Slice(1, 0, 3, 2).Slice(2, 0, 4, 2);
    t.Save(file_name);
    EXPECT_FALSE(t.IsContiguous());
    t_load = core::Tensor::Load(file_name);
    EXPECT_TRUE(t_load.IsContiguous());
    EXPECT_EQ(t_load.GetShape(), core::SizeVector({2, 2, 2}));
    EXPECT_EQ(t_load.ToFlatVector<float>(),
              std::vector<float>({0, 2, 8, 10, 12, 14, 20, 22}));

    // {} tensor (scalar).
    t = core::Tensor::Init<float>(3.14, device);
    t.Save(file_name);
    t_load = core::Tensor::Load(file_name);
    EXPECT_TRUE(t.AllClose(t_load.To(device)));

    // {0} tensor.
    t = core::Tensor::Ones({0}, core::Float32, device);
    t.Save(file_name);
    t_load = core::Tensor::Load(file_name);
    EXPECT_TRUE(t.AllClose(t_load.To(device)));

    // {0, 0} tensor.
    t = core::Tensor::Ones({0, 0}, core::Float32, device);
    t.Save(file_name);
    t_load = core::Tensor::Load(file_name);
    EXPECT_TRUE(t.AllClose(t_load.To(device)));

    // {0, 1, 0} tensor.
    t = core::Tensor::Ones({0, 1, 0}, core::Float32, device);
    t.Save(file_name);
    t_load = core::Tensor::Load(file_name);
    EXPECT_TRUE(t.AllClose(t_load.To(device)));

    // Clean up.
    utility::filesystem::RemoveFile(file_name);
}

}  // namespace tests
}  // namespace open3d
