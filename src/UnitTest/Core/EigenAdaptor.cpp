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
    Eigen::MatrixXd matrix(63, 33);
    matrix.setRandom();

    auto tensor_f64 = FromEigen(matrix);
    auto tensor_f32 = FromEigen(matrix.cast<float>().eval());

    for (int64_t i = 0; i < matrix.rows(); ++i) {
        for (int64_t j = 0; j < matrix.cols(); ++j) {
            double f64_gt = matrix(i, j);
            double f64 = tensor_f64[i][j].Item<double>();
            float f32 = tensor_f32[i][j].Item<float>();
            EXPECT_NEAR(f64, f64_gt, THRESHOLD_1E_6);
            EXPECT_NEAR(f32, f64_gt, THRESHOLD_1E_6);
        }
    }
}

TEST(EigenAdaptor, ToEigen) {
    std::random_device rnd_device;
    std::mt19937 mersenne_engine{rnd_device()};
    std::uniform_real_distribution<float> dist{-10000, 10000};

    int64_t rows = 63;
    int64_t cols = 33;
    std::vector<float> tensor_vals(rows * cols);
    std::generate(tensor_vals.begin(), tensor_vals.end(),
                  [&]() { return dist(mersenne_engine); });

    auto tensor = Tensor(tensor_vals, SizeVector({rows, cols}), Dtype::Float32,
                         Device("CPU:0"));
    auto matrix = ToEigen<float>(tensor);

    for (int64_t i = 0; i < matrix.rows(); ++i) {
        for (int64_t j = 0; j < matrix.cols(); ++j) {
            double fgt = matrix(i, j);
            double f = tensor[i][j].Item<float>();
            EXPECT_NEAR(f, fgt, THRESHOLD_1E_6);
        }
    }
}
}  // namespace unit_test
}  // namespace open3d
