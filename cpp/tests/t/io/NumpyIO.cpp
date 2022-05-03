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

#include "open3d/t/io/NumpyIO.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"
#include "tests/Tests.h"
#include "tests/core/CoreTest.h"

namespace open3d {
namespace tests {

class NumpyIOPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(Tensor,
                         NumpyIOPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

TEST_P(NumpyIOPermuteDevices, NpyWriteRead) {
    const core::Device device = GetParam();
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

TEST_P(NumpyIOPermuteDevices, NpzWriteRead) {
    const core::Device device = GetParam();
    const std::string file_name = "tensors.npz";

    // Empty map.
    t::io::WriteNpz(file_name, {});
    std::unordered_map<std::string, core::Tensor> empty_tensor_map =
            t::io::ReadNpz(file_name);
    EXPECT_EQ(empty_tensor_map.size(), 0);

    core::Tensor t;
    core::Tensor t_load;

    // t0: 2x2 tensor.
    core::Tensor t0 = core::Tensor::Init<int32_t>({{1, 2}, {3, 4}}, device);

    // t1: Non-contiguous tensor will be stored as contiguous tensor.
    // t1 sliced with [0:2:1, 0:3:2, 0:4:2].
    core::Tensor t1 = core::Tensor::Init<float>(
            {{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
             {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}},
            device);
    t1 = t1.Slice(0, 0, 2, 1).Slice(1, 0, 3, 2).Slice(2, 0, 4, 2);

    // t2: {} tensor (scalar).
    core::Tensor t2 = core::Tensor::Init<float>(3.14, device);

    // t3: {0} tensor.
    core::Tensor t3 = core::Tensor::Ones({0}, core::Float32, device);

    // t4: {0, 0} tensor.
    core::Tensor t4 = core::Tensor::Ones({0, 0}, core::Float32, device);

    // t5: {0, 1, 0} tensor.
    core::Tensor t5 = core::Tensor::Ones({0, 1, 0}, core::Float32, device);

    // Write t0 to t5.
    t::io::WriteNpz(file_name, {{"t0", t0},
                                {"t1", t1},
                                {"t2", t2},
                                {"t3", t3},
                                {"t4", t4},
                                {"t5", t5}});

    // Read from npz
    std::unordered_map<std::string, core::Tensor> tensor_map =
            t::io::ReadNpz(file_name);
    EXPECT_EQ(tensor_map.size(), 6);

    core::Tensor t0_load = tensor_map.at("t0");
    EXPECT_TRUE(t0.AllClose(t0_load.To(device)));
    EXPECT_EQ(t0.GetDtype(), t0_load.GetDtype());

    core::Tensor t1_load = tensor_map.at("t1");
    EXPECT_TRUE(t1.AllClose(t1_load.To(device)));
    EXPECT_EQ(t1.GetDtype(), t1_load.GetDtype());

    core::Tensor t2_load = tensor_map.at("t2");
    EXPECT_TRUE(t2.AllClose(t2_load.To(device)));
    EXPECT_EQ(t2.GetDtype(), t2_load.GetDtype());

    core::Tensor t3_load = tensor_map.at("t3");
    EXPECT_TRUE(t3.AllClose(t3_load.To(device)));
    EXPECT_EQ(t3.GetDtype(), t3_load.GetDtype());

    core::Tensor t4_load = tensor_map.at("t4");
    EXPECT_TRUE(t4.AllClose(t4_load.To(device)));
    EXPECT_EQ(t4.GetDtype(), t4_load.GetDtype());

    core::Tensor t5_load = tensor_map.at("t5");
    EXPECT_TRUE(t5.AllClose(t5_load.To(device)));
    EXPECT_EQ(t5.GetDtype(), t5_load.GetDtype());

    // Clean up.
    utility::filesystem::RemoveFile(file_name);
}

}  // namespace tests
}  // namespace open3d
