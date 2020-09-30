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

#include <cmath>
#include <limits>

#include "open3d/core/AdvancedIndexing.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/Kernel.h"
#include "open3d/utility/Helper.h"
#include "tests/UnitTest.h"
#include "tests/core/CoreTest.h"

namespace open3d {
namespace tests {

class LinalgPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(Linalg,
                         LinalgPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

TEST_P(LinalgPermuteDevices, Matmul) {
    const float EPSILON = 1e-8;

    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;

    // Matmul test
    core::Tensor A(std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3}, dtype, device);
    core::Tensor B(
            std::vector<float>{7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},
            {3, 4}, dtype, device);

    core::Tensor C = A.Matmul(B);
    EXPECT_EQ(C.GetShape(), core::SizeVector({2, 4}));
    std::vector<float> C_data = C.ToFlatVector<float>();
    std::vector<float> C_gt = {74, 80, 86, 92, 173, 188, 203, 218};
    for (int i = 0; i < 8; ++i) {
        EXPECT_TRUE(std::abs(C_data[i] - C_gt[i]) < EPSILON);
    }

    // Non-contiguous matmul test
    core::Tensor A_slice = A.GetItem(
            {core::TensorKey::Slice(core::None, core::None, core::None),
             core::TensorKey::Slice(1, core::None, core::None)});
    core::Tensor B_slice =
            B.IndexGet({core::Tensor(std::vector<int64_t>{0, 2}, {2},
                                     core::Dtype::Int64, device)})
                    .GetItem({core::TensorKey::Slice(core::None, core::None,
                                                     core::None)});
    core::Tensor C_slice = A_slice.Matmul(B_slice);

    std::vector<float> C_slice_data = C_slice.ToFlatVector<float>();
    std::vector<float> C_slice_gt = {59, 64, 69, 74, 125, 136, 147, 158};
    for (int i = 0; i < 6; ++i) {
        EXPECT_TRUE(std::abs(C_slice_data[i] - C_slice_gt[i]) < EPSILON);
    }

    // Incompatible shape test
    EXPECT_ANY_THROW(A.Matmul(core::Tensor::Zeros({3, 4, 5}, dtype)));
    EXPECT_ANY_THROW(A.Matmul(core::Tensor::Zeros({3, 0}, dtype)));
    EXPECT_ANY_THROW(A.Matmul(core::Tensor::Zeros({2, 4}, dtype)));
}

TEST_P(LinalgPermuteDevices, Inverse) {
    const float EPSILON = 1e-5;

    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;

    // Inverse test
    core::Tensor A(std::vector<float>{2, 3, 1, 3, 3, 1, 2, 4, 1}, {3, 3}, dtype,
                   device);

    core::Tensor A_inv = A.Inverse();
    EXPECT_EQ(A_inv.GetShape(), core::SizeVector({3, 3}));

    std::vector<float> A_inv_data = A_inv.ToFlatVector<float>();
    std::vector<float> A_inv_gt = {-1, 1, 0, -1, 0, 1, 6, -2, -3};
    for (int i = 0; i < 9; ++i) {
        EXPECT_TRUE(std::abs(A_inv_data[i] - A_inv_gt[i]) < EPSILON);
    }

    // Singular test
    EXPECT_ANY_THROW(core::Tensor::Zeros({3, 3}, dtype, device).Inverse());

    // Shape test
    EXPECT_ANY_THROW(core::Tensor::Ones({0}, dtype, device).Inverse());
    EXPECT_ANY_THROW(core::Tensor::Ones({2, 2, 2}, dtype, device).Inverse());
    EXPECT_ANY_THROW(core::Tensor::Ones({3, 4}, dtype, device).Inverse());
}

TEST_P(LinalgPermuteDevices, SVD) {
    const float EPSILON = 1e-5;

    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;

    // matmul test
    core::Tensor A(std::vector<float>{2, 4, 1, 3, 0, 0, 0, 0}, {4, 2}, dtype,
                   device);

    core::Tensor U, S, VT;
    std::tie(U, S, VT) = A.SVD();
    EXPECT_EQ(U.GetShape(), core::SizeVector({4, 4}));
    EXPECT_EQ(S.GetShape(), core::SizeVector({2}));
    EXPECT_EQ(VT.GetShape(), core::SizeVector({2, 2}));

    core::Tensor UUT = U.Matmul(U.T());
    std::vector<float> UUT_data = UUT.ToFlatVector<float>();
    std::vector<float> UUT_gt = {1, 0, 0, 0, 0, 1, 0, 0,
                                 0, 0, 1, 0, 0, 0, 0, 1};
    for (int i = 0; i < 16; ++i) {
        EXPECT_TRUE(std::abs(UUT_data[i] - UUT_gt[i]) < EPSILON);
    }

    core::Tensor VVT = VT.T().Matmul(VT);
    std::vector<float> VVT_data = VVT.ToFlatVector<float>();
    std::vector<float> VVT_gt = {1, 0, 0, 1};
    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(std::abs(VVT_data[i] - VVT_gt[i]) < EPSILON);
    }

    core::Tensor USVT =
            U.GetItem({core::TensorKey::Slice(core::None, core::None,
                                              core::None),
                       core::TensorKey::Slice(core::None, 2, core::None)})
                    .Matmul(core::Tensor::Diag(S).Matmul(VT));
    EXPECT_EQ(USVT.GetShape(), A.GetShape());

    std::vector<float> A_data = A.ToFlatVector<float>();
    std::vector<float> USVT_data = USVT.ToFlatVector<float>();
    for (int i = 0; i < 8; ++i) {
        EXPECT_TRUE(std::abs(A_data[i] - USVT_data[i]) < EPSILON);
    }
}

TEST_P(LinalgPermuteDevices, Solve) {
    const float EPSILON = 1e-8;

    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;

    // Solve test
    core::Tensor A(std::vector<float>{3, 1, 1, 2}, {2, 2}, dtype, device);
    core::Tensor B(std::vector<float>{9, 8}, {2}, dtype, device);
    core::Tensor X = A.Solve(B);

    EXPECT_EQ(X.GetShape(), core::SizeVector{2});
    std::vector<float> X_data = X.ToFlatVector<float>();
    std::vector<float> X_gt = std::vector<float>{2, 3};
    for (int i = 0; i < 2; ++i) {
        EXPECT_TRUE(std::abs(X_data[i] - X_gt[i]) < EPSILON);
    }

    // Singular test
    EXPECT_ANY_THROW(core::Tensor::Zeros({2, 2}, dtype, device).Solve(B));

    // Shape test
    EXPECT_ANY_THROW(core::Tensor::Ones({2, 3}, dtype, device).Solve(B));
    EXPECT_ANY_THROW(core::Tensor::Ones({2, 2, 2}, dtype, device).Solve(B));
    EXPECT_ANY_THROW(core::Tensor::Ones({2, 0}, dtype, device).Solve(B));
    EXPECT_ANY_THROW(core::Tensor::Ones({2}, dtype, device).Solve(B));
}

TEST_P(LinalgPermuteDevices, LeastSquares) {
    const float EPSILON = 1e-5;

    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;

    // Solve test
    core::Tensor A(std::vector<float>{1.44,  -7.84, -4.39, 4.53,  -9.96, -0.28,
                                      -3.24, 3.83,  -7.55, 3.24,  6.27,  -6.64,
                                      8.34,  8.09,  5.28,  2.06,  7.08,  2.52,
                                      0.74,  -2.47, -5.45, -5.70, -1.19, 4.70},
                   {6, 4}, dtype, device);
    core::Tensor B(std::vector<float>{8.58, 9.35, 8.26, -4.43, 8.48, -0.70,
                                      -5.28, -0.26, 5.72, -7.36, 8.93, -2.52},
                   {6, 2}, dtype, device);
    core::Tensor X = A.LeastSquares(B);

    EXPECT_EQ(X.GetShape(), core::SizeVector({4, 2}));

    std::vector<float> X_data = X.ToFlatVector<float>();
    std::vector<float> X_gt = std::vector<float>{
            -0.45063714, 0.249748,   -0.84915021, -0.90201926,
            0.70661216,  0.63234303, 0.12888575,  0.13512364};
    for (int i = 0; i < 2; ++i) {
        EXPECT_TRUE(std::abs(X_data[i] - X_gt[i]) < EPSILON);
    }
}
}  // namespace tests
}  // namespace open3d
