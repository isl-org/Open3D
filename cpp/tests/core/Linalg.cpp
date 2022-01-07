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

#include <cmath>
#include <limits>

#include "open3d/core/AdvancedIndexing.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/Kernel.h"
#include "open3d/core/linalg/AddMM.h"
#include "open3d/core/linalg/kernel/SVD3x3.h"
#include "open3d/utility/Helper.h"
#include "tests/Tests.h"
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
    core::Dtype dtype = core::Float32;

    // Matmul test.
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

    // Non-contiguous matmul test.
    core::Tensor A_slice = A.GetItem(
            {core::TensorKey::Slice(core::None, core::None, core::None),
             core::TensorKey::Slice(1, core::None, core::None)});
    core::Tensor B_slice =
            B.IndexGet({core::Tensor(std::vector<int64_t>{0, 2}, {2},
                                     core::Int64, device)})
                    .GetItem({core::TensorKey::Slice(core::None, core::None,
                                                     core::None)});
    core::Tensor C_slice = A_slice.Matmul(B_slice);

    std::vector<float> C_slice_data = C_slice.ToFlatVector<float>();
    std::vector<float> C_slice_gt = {59, 64, 69, 74, 125, 136, 147, 158};
    for (int i = 0; i < 6; ++i) {
        EXPECT_TRUE(std::abs(C_slice_data[i] - C_slice_gt[i]) < EPSILON);
    }

    // Incompatible shape test.
    EXPECT_ANY_THROW(A.Matmul(core::Tensor::Zeros({3, 4, 5}, dtype)));
    EXPECT_ANY_THROW(A.Matmul(core::Tensor::Zeros({3, 0}, dtype)));
    EXPECT_ANY_THROW(A.Matmul(core::Tensor::Zeros({2, 4}, dtype)));
}

TEST_P(LinalgPermuteDevices, AddMM) {
    const float EPSILON = 1e-8;

    core::Device device = GetParam();
    core::Dtype dtype = core::Float32;

    // addmm test.
    core::Tensor A = core::Tensor::Init<float>({{1, 2, 3}, {4, 5, 6}}, device);
    core::Tensor B = core::Tensor::Init<float>(
            {{7, 8, 9, 10}, {11, 12, 13, 14}, {15, 16, 17, 18}}, device);
    core::Tensor B_T = B.T().Contiguous();

    core::Tensor C = core::Tensor::Ones({2, 4}, dtype, device);
    core::AddMM(A, B, C, 1.0, 1.0);
    EXPECT_EQ(C.GetShape(), core::SizeVector({2, 4}));
    core::Tensor C_gt = core::Tensor::Init<float>(
            {{75, 81, 87, 93}, {174, 189, 204, 219}}, device);
    EXPECT_TRUE(C_gt.AllClose(C, EPSILON));

    // alpha = -2.0 & beta = 5.0.
    C = core::Tensor::Ones({2, 4}, dtype, device);
    core::AddMM(A, B, C, -2.0, 5.0);
    EXPECT_EQ(C.GetShape(), core::SizeVector({2, 4}));
    C_gt = core::Tensor::Init<float>(
            {{-143, -155, -167, -179}, {-341, -371, -401, -431}}, device);
    EXPECT_TRUE(C_gt.AllClose(C, EPSILON));

    // Transposed addmm test.
    C = core::Tensor::Ones({2, 4}, dtype, device);
    core::Tensor B_T_T = B_T.T();
    core::AddMM(A, B_T_T, C, 1.0, 1.0);
    EXPECT_EQ(C.GetShape(), core::SizeVector({2, 4}));
    C_gt = core::Tensor::Init<float>({{75, 81, 87, 93}, {174, 189, 204, 219}},
                                     device);
    EXPECT_TRUE(C_gt.AllClose(C, EPSILON));

    // Transposed addmm + alpha = -2.0 & beta = 5.0.
    C = core::Tensor::Ones({2, 4}, dtype, device);
    B_T_T = B_T.T();
    core::AddMM(A, B_T_T, C, -2.0, 5.0);
    EXPECT_EQ(C.GetShape(), core::SizeVector({2, 4}));
    C_gt = core::Tensor::Init<float>(
            {{-143, -155, -167, -179}, {-341, -371, -401, -431}}, device);
    EXPECT_TRUE(C_gt.AllClose(C, EPSILON));

    // Non-contiguous addmm test.
    core::Tensor A_slice = A.GetItem(
            {core::TensorKey::Slice(core::None, core::None, core::None),
             core::TensorKey::Slice(1, core::None, core::None)});
    core::Tensor B_slice =
            B.IndexGet({core::Tensor(std::vector<int64_t>{0, 2}, {2},
                                     core::Int64, device)})
                    .GetItem({core::TensorKey::Slice(core::None, core::None,
                                                     core::None)});

    C = core::Tensor::Ones({2, 4}, dtype, device);
    core::AddMM(A_slice, B_slice, C, 1.0, 1.0);
    EXPECT_EQ(C.GetShape(), core::SizeVector({2, 4}));
    C_gt = core::Tensor::Init<float>({{60, 65, 70, 75}, {126, 137, 148, 159}},
                                     device);
    EXPECT_TRUE(C_gt.AllClose(C, EPSILON));
}

TEST_P(LinalgPermuteDevices, LU) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Float32;

    // LU test for 3x4 const 2D tensor of dtype Float32.
    const core::Tensor A_3x4cf = core::Tensor::Init<float>(
            {{2, 3, 1}, {3, 3, 1}, {2, 4, 1}, {2, 1, 3}}, device);

    // Default parameter for permute_l (false).
    core::Tensor permutationcf0, lowercf0, uppercf0;
    std::tie(permutationcf0, lowercf0, uppercf0) = A_3x4cf.LU();
    core::Tensor outputcf0 =
            permutationcf0.Matmul(lowercf0.Matmul(uppercf0)).Contiguous();
    EXPECT_TRUE(A_3x4cf.AllClose(outputcf0, FLT_EPSILON, FLT_EPSILON));

    // "permute_l = true": L must be P*L. So, output = L*U.
    core::Tensor permutationcf1, lowercf1, uppercf1;
    std::tie(permutationcf1, lowercf1, uppercf1) =
            A_3x4cf.LU(/*permute_l=*/true);
    core::Tensor outputcf1 = lowercf1.Matmul(uppercf1).Contiguous();
    EXPECT_TRUE(A_3x4cf.AllClose(outputcf1, FLT_EPSILON, FLT_EPSILON));

    // LU test for 3x3 const square 2D tensor of dtype Float64.
    const core::Tensor A_3x3cd = core::Tensor::Init<double>(
            {{2, 3, 1}, {3, 3, 1}, {2, 4, 1}}, device);
    core::Tensor permutationcd0, lowercd0, uppercd0;
    std::tie(permutationcd0, lowercd0, uppercd0) = A_3x3cd.LU();
    core::Tensor outputcd0 =
            permutationcd0.Matmul(lowercd0.Matmul(uppercd0)).Contiguous();
    EXPECT_TRUE(A_3x3cd.AllClose(outputcd0, DBL_EPSILON, DBL_EPSILON));

    // Singular test.
    EXPECT_ANY_THROW(core::Tensor::Zeros({3, 3}, dtype, device).LU());

    // Shape test.
    EXPECT_ANY_THROW(core::Tensor::Ones({0}, dtype, device).LU());
    EXPECT_ANY_THROW(core::Tensor::Ones({2, 2, 2}, dtype, device).LU());
}

TEST_P(LinalgPermuteDevices, LUIpiv) {
    const float EPSILON = 1e-6;
    core::Device device = GetParam();
    core::Dtype dtype = core::Float32;

    // LU test for 3x3 square 2D tensor of dtype Float32.
    const core::Tensor A_3x3f = core::Tensor::Init<float>(
            {{2, 3, 1}, {3, 3, 1}, {2, 4, 1}}, device);

    core::Tensor ipiv3f, A3f;
    std::tie(ipiv3f, A3f) = A_3x3f.LUIpiv();

    EXPECT_TRUE(
            A3f.AllClose(core::Tensor::Init<float>({{3.0, 3.0, 1.0},
                                                    {0.666667, 2.0, 0.333333},
                                                    {0.666667, 0.5, 0.166667}},
                                                   device),
                         EPSILON, EPSILON));
    EXPECT_TRUE(ipiv3f.To(core::Int32)
                        .AllClose(core::Tensor::Init<int>({2, 3, 3}, device),
                                  EPSILON));

    // LU test for 3x3 square 2D tensor of dtype Float64.
    core::Tensor A_3x3d = core::Tensor::Init<double>(
            {{2, 3, 1}, {3, 3, 1}, {2, 4, 1}}, device);

    core::Tensor ipiv3d, A3d;
    std::tie(ipiv3d, A3d) = A_3x3d.LUIpiv();

    EXPECT_TRUE(
            A3d.AllClose(core::Tensor::Init<double>({{3.0, 3.0, 1.0},
                                                     {0.666667, 2.0, 0.333333},
                                                     {0.666667, 0.5, 0.166667}},
                                                    device),
                         EPSILON, EPSILON));
    EXPECT_TRUE(ipiv3d.To(core::Int32)
                        .AllClose(core::Tensor::Init<int>({2, 3, 3}, device),
                                  EPSILON));

    // Singular test.
    EXPECT_ANY_THROW(core::Tensor::Zeros({3, 3}, dtype, device).LU());

    // Shape test.
    EXPECT_ANY_THROW(core::Tensor::Ones({0}, dtype, device).LU());
    EXPECT_ANY_THROW(core::Tensor::Ones({2, 2, 2}, dtype, device).LU());
}

TEST_P(LinalgPermuteDevices, Triu) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Float32;

    // Input 2D matrix of dtype Float32.
    const core::Tensor A_4x5f =
            core::Tensor::Init<float>({{1, 2, 3, 4, 5},
                                       {6, 7, 8, 9, 10},
                                       {11, 12, 13, 14, 15},
                                       {16, 17, 18, 19, 20}},
                                      device);

    // Get upper triangle matrix from main diagonal (= 0).
    core::Tensor Uf0 = A_4x5f.Triu();
    EXPECT_TRUE(Uf0.AllClose(core::Tensor::Init<float>({{1, 2, 3, 4, 5},
                                                        {0, 7, 8, 9, 10},
                                                        {0, 0, 13, 14, 15},
                                                        {0, 0, 0, 19, 20}},
                                                       device)));

    // Get upper triangle matrix from diagonal (= 1).
    core::Tensor Uf1 = A_4x5f.Triu(1);
    EXPECT_TRUE(Uf1.AllClose(core::Tensor::Init<float>({{0, 2, 3, 4, 5},
                                                        {0, 0, 8, 9, 10},
                                                        {0, 0, 0, 14, 15},
                                                        {0, 0, 0, 0, 20}},
                                                       device)));

    // Get upper triangle matrix from diagonal (= -1).
    core::Tensor Uf1_ = A_4x5f.Triu(-1);
    EXPECT_TRUE(Uf1_.AllClose(core::Tensor::Init<float>({{1, 2, 3, 4, 5},
                                                         {6, 7, 8, 9, 10},
                                                         {0, 12, 13, 14, 15},
                                                         {0, 0, 18, 19, 20}},
                                                        device)));

    // Boundary test.
    EXPECT_ANY_THROW(A_4x5f.Triu(-4));
    EXPECT_ANY_THROW(A_4x5f.Triu(5));

    // Shape test.
    EXPECT_ANY_THROW(core::Tensor::Ones({0}, dtype, device).Triu());
    EXPECT_ANY_THROW(core::Tensor::Ones({2, 2, 2}, dtype, device).Triu());
}

TEST_P(LinalgPermuteDevices, Tril) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Float32;

    // Input 2D matrix of dtype Float32.
    const core::Tensor A_4x5f =
            core::Tensor::Init<float>({{1, 2, 3, 4, 5},
                                       {6, 7, 8, 9, 10},
                                       {11, 12, 13, 14, 15},
                                       {16, 17, 18, 19, 20}},
                                      device);

    // Get lower triangle matrix from main diagonal (= 0).
    core::Tensor Lf0 = A_4x5f.Tril();
    EXPECT_TRUE(Lf0.AllClose(core::Tensor::Init<float>({{1, 0, 0, 0, 0},
                                                        {6, 7, 0, 0, 0},
                                                        {11, 12, 13, 0, 0},
                                                        {16, 17, 18, 19, 0}},
                                                       device)));

    // Get lower triangle matrix from diagonal (= 1).
    core::Tensor Lf1 = A_4x5f.Tril(1);
    EXPECT_TRUE(Lf1.AllClose(core::Tensor::Init<float>({{1, 2, 0, 0, 0},
                                                        {6, 7, 8, 0, 0},
                                                        {11, 12, 13, 14, 0},
                                                        {16, 17, 18, 19, 20}},
                                                       device)));

    // Get lower triangle matrix from diagonal (= -1).
    core::Tensor Lf1_ = A_4x5f.Tril(-1);
    EXPECT_TRUE(Lf1_.AllClose(core::Tensor::Init<float>({{0, 0, 0, 0, 0},
                                                         {6, 0, 0, 0, 0},
                                                         {11, 12, 0, 0, 0},
                                                         {16, 17, 18, 0, 0}},
                                                        device)));

    core::Tensor Lf4 = A_4x5f.Tril(4);
    EXPECT_TRUE(Lf4.AllClose(core::Tensor::Init<float>({{1, 2, 3, 4, 5},
                                                        {6, 7, 8, 9, 10},
                                                        {11, 12, 13, 14, 15},
                                                        {16, 17, 18, 19, 20}},
                                                       device)));

    // Boundary test.
    EXPECT_ANY_THROW(A_4x5f.Tril(-5));
    EXPECT_ANY_THROW(A_4x5f.Tril(6));

    // Shape test.
    EXPECT_ANY_THROW(core::Tensor::Ones({0}, dtype, device).Tril());
    EXPECT_ANY_THROW(core::Tensor::Ones({2, 2, 2}, dtype, device).Tril());
}

TEST_P(LinalgPermuteDevices, Triul) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Float32;

    // Input 2D matrix of dtype Float32.
    const core::Tensor A_4x5f =
            core::Tensor::Init<float>({{1, 2, 3, 4, 5},
                                       {6, 7, 8, 9, 10},
                                       {11, 12, 13, 14, 15},
                                       {16, 17, 18, 19, 20}},
                                      device);

    // Get lower triangle matrix from main diagonal (= 0).
    core::Tensor Uf0, Lf0;
    std::tie(Uf0, Lf0) = A_4x5f.Triul();
    EXPECT_TRUE(Uf0.AllClose(core::Tensor::Init<float>({{1, 2, 3, 4, 5},
                                                        {0, 7, 8, 9, 10},
                                                        {0, 0, 13, 14, 15},
                                                        {0, 0, 0, 19, 20}},
                                                       device)));
    EXPECT_TRUE(Lf0.AllClose(core::Tensor::Init<float>({{1, 0, 0, 0, 0},
                                                        {6, 1, 0, 0, 0},
                                                        {11, 12, 1, 0, 0},
                                                        {16, 17, 18, 1, 0}},
                                                       device)));

    core::Tensor Uf1, Lf1;
    std::tie(Uf1, Lf1) = A_4x5f.Triul(1);
    EXPECT_TRUE(Uf1.AllClose(core::Tensor::Init<float>({{0, 2, 3, 4, 5},
                                                        {0, 0, 8, 9, 10},
                                                        {0, 0, 0, 14, 15},
                                                        {0, 0, 0, 0, 20}},
                                                       device)));
    EXPECT_TRUE(Lf1.AllClose(core::Tensor::Init<float>({{1, 1, 0, 0, 0},
                                                        {6, 7, 1, 0, 0},
                                                        {11, 12, 13, 1, 0},
                                                        {16, 17, 18, 19, 1}},
                                                       device)));

    core::Tensor Uf1_, Lf1_;
    std::tie(Uf1_, Lf1_) = A_4x5f.Triul(-1);
    EXPECT_TRUE(Uf1_.AllClose(core::Tensor::Init<float>({{1, 2, 3, 4, 5},
                                                         {6, 7, 8, 9, 10},
                                                         {0, 12, 13, 14, 15},
                                                         {0, 0, 18, 19, 20}},
                                                        device)));
    EXPECT_TRUE(Lf1_.AllClose(core::Tensor::Init<float>({{0, 0, 0, 0, 0},
                                                         {1, 0, 0, 0, 0},
                                                         {11, 1, 0, 0, 0},
                                                         {16, 17, 1, 0, 0}},
                                                        device)));

    // Boundary test.
    EXPECT_ANY_THROW(A_4x5f.Triul(-4));
    EXPECT_ANY_THROW(A_4x5f.Triul(5));

    // Shape test.
    EXPECT_ANY_THROW(core::Tensor::Ones({0}, dtype, device).Triul());
    EXPECT_ANY_THROW(core::Tensor::Ones({2, 2, 2}, dtype, device).Triul());
}

TEST_P(LinalgPermuteDevices, Inverse) {
    const float EPSILON = 1e-5;

    core::Device device = GetParam();
    core::Dtype dtype = core::Float32;

    // Inverse test.
    core::Tensor A(std::vector<float>{2, 3, 1, 3, 3, 1, 2, 4, 1}, {3, 3}, dtype,
                   device);

    core::Tensor A_inv = A.Inverse();
    EXPECT_EQ(A_inv.GetShape(), core::SizeVector({3, 3}));

    std::vector<float> A_inv_data = A_inv.ToFlatVector<float>();
    std::vector<float> A_inv_gt = {-1, 1, 0, -1, 0, 1, 6, -2, -3};
    for (int i = 0; i < 9; ++i) {
        EXPECT_TRUE(std::abs(A_inv_data[i] - A_inv_gt[i]) < EPSILON);
    }

    // Singular test.
    EXPECT_ANY_THROW(core::Tensor::Zeros({3, 3}, dtype, device).Inverse());

    // Shape test.
    EXPECT_ANY_THROW(core::Tensor::Ones({0}, dtype, device).Inverse());
    EXPECT_ANY_THROW(core::Tensor::Ones({2, 2, 2}, dtype, device).Inverse());
    EXPECT_ANY_THROW(core::Tensor::Ones({3, 4}, dtype, device).Inverse());
}

TEST_P(LinalgPermuteDevices, SVD) {
    const float EPSILON = 1e-5;

    core::Device device = GetParam();
    core::Dtype dtype = core::Float32;

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
    core::Dtype dtype = core::Float32;

    // Solve test.
    core::Tensor A(std::vector<float>{3, 1, 1, 2}, {2, 2}, dtype, device);
    core::Tensor B(std::vector<float>{9, 8}, {2}, dtype, device);
    core::Tensor X = A.Solve(B);

    EXPECT_EQ(X.GetShape(), core::SizeVector{2});
    std::vector<float> X_data = X.ToFlatVector<float>();
    std::vector<float> X_gt = std::vector<float>{2, 3};
    for (int i = 0; i < 2; ++i) {
        EXPECT_TRUE(std::abs(X_data[i] - X_gt[i]) < EPSILON);
    }

    // Singular test.
    EXPECT_ANY_THROW(core::Tensor::Zeros({2, 2}, dtype, device).Solve(B));

    // Shape test.
    EXPECT_ANY_THROW(core::Tensor::Ones({2, 3}, dtype, device).Solve(B));
    EXPECT_ANY_THROW(core::Tensor::Ones({2, 2, 2}, dtype, device).Solve(B));
    EXPECT_ANY_THROW(core::Tensor::Ones({2, 0}, dtype, device).Solve(B));
    EXPECT_ANY_THROW(core::Tensor::Ones({2}, dtype, device).Solve(B));
}

TEST_P(LinalgPermuteDevices, LeastSquares) {
    const float EPSILON = 1e-5;

    core::Device device = GetParam();
    core::Dtype dtype = core::Float32;

    // Solve test.
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

TEST_P(LinalgPermuteDevices, KernelOps) {
    core::Tensor A_3x3 =
            core::Tensor::Init<float>({{0, 1, 0}, {1, 0, 0}, {0, 0, 1}});
    core::Tensor B_3x1 = core::Tensor::Init<float>({{1}, {3}, {6}});
    core::Tensor I_3x3 =
            core::Tensor::Eye(3, core::Float32, core::Device("CPU:0"));

    core::Tensor output3x3 =
            core::Tensor::Empty({3, 3}, core::Float32, core::Device("CPU:0"));
    core::Tensor output3x1 =
            core::Tensor::Empty({3, 1}, core::Float32, core::Device("CPU:0"));

    // {3, 3} x {3, 3} MatMul
    auto matmul3x3_expected = A_3x3.Matmul(I_3x3);

    core::linalg::kernel::matmul3x3_3x3(A_3x3.GetDataPtr<float>(),
                                        I_3x3.GetDataPtr<float>(),
                                        output3x3.GetDataPtr<float>());

    EXPECT_TRUE(output3x3.AllClose(matmul3x3_expected));

    // {3, 3} x {3, 1} MatMul
    auto matmul3x1_expected = A_3x3.Matmul(B_3x1);

    core::linalg::kernel::matmul3x3_3x1(A_3x3.GetDataPtr<float>(),
                                        B_3x1.GetDataPtr<float>(),
                                        output3x1.GetDataPtr<float>());
    EXPECT_TRUE(output3x1.AllClose(matmul3x1_expected));

    // Inverse 3x3
    auto Ainv_expected = A_3x3.Inverse();
    core::linalg::kernel::inverse3x3(A_3x3.GetDataPtr<float>(),
                                     output3x3.GetDataPtr<float>());
    EXPECT_TRUE(output3x3.AllClose(Ainv_expected));

    // Transpose 3x3
    auto AT_expected = A_3x3.T();
    core::linalg::kernel::transpose3x3(A_3x3.GetDataPtr<float>(),
                                       output3x3.GetDataPtr<float>());
    EXPECT_TRUE(output3x3.AllClose(AT_expected));

    // Det 3x3
    double det_expected = A_3x3.Det();
    double det_output = static_cast<double>(
            core::linalg::kernel::det3x3(A_3x3.GetDataPtr<float>()));
    EXPECT_EQ(det_output, det_expected);

    // SVD Solver 3x3.
    core::linalg::kernel::solve_svd3x3(A_3x3.GetDataPtr<float>(),
                                       B_3x1.GetDataPtr<float>(),
                                       output3x1.GetDataPtr<float>());
    auto Solve_Expected = A_3x3.Solve(B_3x1);
    EXPECT_TRUE(output3x1.AllClose(Solve_Expected));
}

}  // namespace tests
}  // namespace open3d
