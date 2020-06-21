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

#include "open3d/core/Indexer.h"
#include "open3d/core/Device.h"
#include "open3d/core/SizeVector.h"

#include <unordered_map>

#include "tests/UnitTest.h"
#include "tests/core/CoreTest.h"

namespace open3d {
namespace tests {

class IndexerPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(Indexer,
                         IndexerPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

class IndexerPermuteDevicePairs : public PermuteDevicePairs {};
INSTANTIATE_TEST_SUITE_P(
        Indexer,
        IndexerPermuteDevicePairs,
        testing::ValuesIn(IndexerPermuteDevicePairs::TestCases()));

class IndexerPermuteSizesDefaultStridesAndDevices
    : public testing::TestWithParam<
              std::tuple<std::pair<core::SizeVector, core::SizeVector>,
                         core::Device>> {};
INSTANTIATE_TEST_SUITE_P(
        Indexer,
        IndexerPermuteSizesDefaultStridesAndDevices,
        testing::Combine(
                testing::ValuesIn(PermuteSizesDefaultStrides::TestCases()),
                testing::ValuesIn(PermuteDevices::TestCases())));

TEST_P(IndexerPermuteDevices, TensorRef) {
    core::Device device = GetParam();

    core::Tensor t({2, 1, 3}, core::Dtype::Float32, device);
    core::TensorRef tr(t);

    EXPECT_EQ(tr.ndims_, 3);
    EXPECT_EQ(tr.dtype_byte_size_, 4);
    EXPECT_EQ(tr.data_ptr_, t.GetDataPtr());
    EXPECT_EQ(core::SizeVector(tr.shape_, tr.shape_ + 3),
              core::SizeVector({2, 1, 3}));
    EXPECT_EQ(core::SizeVector(tr.byte_strides_, tr.byte_strides_ + 3),
              core::SizeVector({3 * 4, 3 * 4, 1 * 4}));

    // Test default copy constructor.
    core::TensorRef tr_new = tr;
    EXPECT_EQ(tr_new.ndims_, tr.ndims_);
    EXPECT_EQ(tr_new.dtype_byte_size_, tr.dtype_byte_size_);
    EXPECT_EQ(tr_new.data_ptr_, tr.data_ptr_);
    EXPECT_EQ(core::SizeVector(tr_new.shape_, tr_new.shape_ + 3),
              core::SizeVector({2, 1, 3}));
    EXPECT_EQ(core::SizeVector(tr_new.byte_strides_, tr_new.byte_strides_ + 3),
              core::SizeVector({3 * 4, 3 * 4, 1 * 4}));
}

TEST_P(IndexerPermuteDevices, IndexerCopyConstructor) {
    core::Device device = GetParam();

    core::Tensor input0({2, 1, 1, 3}, core::Dtype::Float32, device);
    core::Tensor input1({1, 3}, core::Dtype::Float32, device);
    core::Tensor output({2, 2, 2, 1, 3}, core::Dtype::Float32, device);
    core::Indexer indexer_a({input0, input1}, output);
    core::Indexer indexer_b = indexer_a;

    EXPECT_EQ(indexer_a.NumInputs(), indexer_b.NumInputs());
    EXPECT_EQ(indexer_a.GetInput(0), indexer_b.GetInput(0));
    EXPECT_EQ(indexer_a.GetInput(1), indexer_b.GetInput(1));
    EXPECT_EQ(indexer_a.GetOutput(), indexer_b.GetOutput());
    EXPECT_EQ(indexer_a.NumDims(), indexer_b.NumDims());
    for (int64_t i = 0; i < indexer_a.NumDims(); i++) {
        EXPECT_EQ(indexer_a.GetMasterShape()[i], indexer_b.GetMasterShape()[i]);
        EXPECT_EQ(indexer_a.GetMasterStrides()[i],
                  indexer_b.GetMasterStrides()[i]);
        EXPECT_EQ(indexer_a.IsReductionDim(i), indexer_b.IsReductionDim(i));
    }
}

TEST_P(IndexerPermuteDevices, BroadcastRestride) {
    core::Device device = GetParam();

    core::Tensor input0({2, 1, 1, 3}, core::Dtype::Float32, device);
    core::Tensor input1({1, 3}, core::Dtype::Float32, device);
    core::Tensor output({2, 2, 2, 1, 3}, core::Dtype::Float32, device);
    core::Indexer indexer({input0, input1}, output);

    core::TensorRef input0_tr = indexer.GetInput(0);
    core::TensorRef input1_tr = indexer.GetInput(1);
    core::TensorRef output_tr = indexer.GetOutput();

    EXPECT_EQ(input0_tr.ndims_, 5);
    EXPECT_EQ(input1_tr.ndims_, 5);
    EXPECT_EQ(output_tr.ndims_, 5);

    // Check core::Indexer's global info
    EXPECT_EQ(indexer.NumInputs(), 2);
    EXPECT_EQ(indexer.NumWorkloads(), 24);
    EXPECT_EQ(core::SizeVector(indexer.GetMasterShape(),
                               indexer.GetMasterShape() + indexer.NumDims()),
              core::SizeVector({2, 2, 2, 1, 3}));
    EXPECT_EQ(core::SizeVector(indexer.GetMasterStrides(),
                               indexer.GetMasterStrides() + indexer.NumDims()),
              core::SizeVector({12, 6, 3, 3, 1}));

    // Check tensor shape
    EXPECT_EQ(core::SizeVector(input0_tr.shape_,
                               input0_tr.shape_ + input0_tr.ndims_),
              core::SizeVector({1, 2, 1, 1, 3}));
    EXPECT_EQ(core::SizeVector(input1_tr.shape_,
                               input1_tr.shape_ + input1_tr.ndims_),
              core::SizeVector({1, 1, 1, 1, 3}));
    EXPECT_EQ(core::SizeVector(output_tr.shape_,
                               output_tr.shape_ + output_tr.ndims_),
              core::SizeVector({2, 2, 2, 1, 3}));

    // Check tensor strides
    EXPECT_EQ(core::SizeVector(input0_tr.byte_strides_,
                               input0_tr.byte_strides_ + input0_tr.ndims_),
              core::SizeVector({0, 3 * 4, 0, 3 * 4, 1 * 4}));
    EXPECT_EQ(core::SizeVector(input1_tr.byte_strides_,
                               input1_tr.byte_strides_ + input1_tr.ndims_),
              core::SizeVector({0, 0, 0, 3 * 4, 1 * 4}));
    EXPECT_EQ(core::SizeVector(output_tr.byte_strides_,
                               output_tr.byte_strides_ + output_tr.ndims_),
              core::SizeVector({12 * 4, 6 * 4, 3 * 4, 3 * 4, 1 * 4}));
}

TEST_P(IndexerPermuteDevices, GetPointers) {
    core::Device device = GetParam();

    core::Tensor input0({3, 1, 1}, core::Dtype::Float32, device);
    core::Tensor input1({2, 1}, core::Dtype::Float32, device);
    core::Tensor output({3, 2, 1}, core::Dtype::Float32, device);
    core::Indexer indexer({input0, input1}, output);

    char* input0_base_ptr = static_cast<char*>(input0.GetDataPtr());
    char* input1_base_ptr = static_cast<char*>(input1.GetDataPtr());
    char* output_base_ptr = static_cast<char*>(output.GetDataPtr());
    int64_t dtype_byte_size = core::DtypeUtil::ByteSize(core::Dtype::Float32);

    EXPECT_EQ(indexer.GetInputPtr(0, 0), input0_base_ptr + 0 * dtype_byte_size);
    EXPECT_EQ(indexer.GetInputPtr(0, 1), input0_base_ptr + 0 * dtype_byte_size);
    EXPECT_EQ(indexer.GetInputPtr(0, 2), input0_base_ptr + 1 * dtype_byte_size);
    EXPECT_EQ(indexer.GetInputPtr(0, 3), input0_base_ptr + 1 * dtype_byte_size);
    EXPECT_EQ(indexer.GetInputPtr(0, 4), input0_base_ptr + 2 * dtype_byte_size);
    EXPECT_EQ(indexer.GetInputPtr(0, 5), input0_base_ptr + 2 * dtype_byte_size);

    EXPECT_EQ(indexer.GetInputPtr(1, 0), input1_base_ptr + 0 * dtype_byte_size);
    EXPECT_EQ(indexer.GetInputPtr(1, 1), input1_base_ptr + 1 * dtype_byte_size);
    EXPECT_EQ(indexer.GetInputPtr(1, 2), input1_base_ptr + 0 * dtype_byte_size);
    EXPECT_EQ(indexer.GetInputPtr(1, 3), input1_base_ptr + 1 * dtype_byte_size);
    EXPECT_EQ(indexer.GetInputPtr(1, 4), input1_base_ptr + 0 * dtype_byte_size);
    EXPECT_EQ(indexer.GetInputPtr(1, 5), input1_base_ptr + 1 * dtype_byte_size);

    EXPECT_EQ(indexer.GetOutputPtr(0), output_base_ptr + 0 * dtype_byte_size);
    EXPECT_EQ(indexer.GetOutputPtr(1), output_base_ptr + 1 * dtype_byte_size);
    EXPECT_EQ(indexer.GetOutputPtr(2), output_base_ptr + 2 * dtype_byte_size);
    EXPECT_EQ(indexer.GetOutputPtr(3), output_base_ptr + 3 * dtype_byte_size);
    EXPECT_EQ(indexer.GetOutputPtr(4), output_base_ptr + 4 * dtype_byte_size);
    EXPECT_EQ(indexer.GetOutputPtr(5), output_base_ptr + 5 * dtype_byte_size);
}

}  // namespace tests
}  // namespace open3d
