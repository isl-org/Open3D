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

#include <random>

#include "Open3D/Core/EigenAdaptor.h"
#include "TestUtility/UnitTest.h"

namespace open3d {
namespace unit_test {
TEST(EigenAdaptor, FromEigen) {
    Eigen::MatrixXd matrix(2, 3);
    matrix << 0, 1, 2, 3, 4, 5;

    Tensor tensor_f64 = FromEigen(matrix);
    EXPECT_EQ(tensor_f64.GetShape(), SizeVector({2, 3}));
    EXPECT_EQ(tensor_f64.GetDevice(), Device("CPU:0"));
    EXPECT_EQ(tensor_f64.GetDtype(), Dtype::Float64);
    EXPECT_EQ(tensor_f64.ToFlatVector<double>(),
              std::vector<double>({0, 1, 2, 3, 4, 5}));

    Tensor tensor_f32 = FromEigen(matrix.cast<float>().eval());
    EXPECT_EQ(tensor_f32.GetShape(), SizeVector({2, 3}));
    EXPECT_EQ(tensor_f32.GetDevice(), Device("CPU:0"));
    EXPECT_EQ(tensor_f32.GetDtype(), Dtype::Float32);
    EXPECT_EQ(tensor_f32.ToFlatVector<float>(),
              std::vector<float>({0, 1, 2, 3, 4, 5}));
}

TEST(EigenAdaptor, ToEigen) {
    Tensor t(std::vector<float>({0, 1, 2, 3, 4, 5}), {2, 3}, Dtype::Float32);

    Eigen::Matrix<float, -1, -1, Eigen::RowMajor> ref_row_major(2, 3);
    ref_row_major << 0, 1, 2, 3, 4, 5;
    Eigen::Matrix<float, -1, -1, Eigen::RowMajor> row_major =
            ToEigen<float>(t, Eigen::RowMajor);
    ExpectEQ(row_major, ref_row_major);
    EXPECT_EQ(row_major.rows(), ref_row_major.rows());
    EXPECT_EQ(row_major.cols(), ref_row_major.cols());

    Eigen::Matrix<float, -1, -1, Eigen::ColMajor> ref_col_major(2, 3);
    ref_col_major << 0, 1, 2, 3, 4, 5;
    Eigen::Matrix<float, -1, -1, Eigen::ColMajor> col_major =
            ToEigen<float>(t, Eigen::ColMajor);
    ExpectEQ(col_major, ref_col_major);
    EXPECT_EQ(col_major.rows(), ref_col_major.rows());
    EXPECT_EQ(col_major.cols(), ref_col_major.cols());
}
}  // namespace unit_test
}  // namespace open3d
