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

#include "open3d/Open3D.h"

#include "open3d/core/op/linalg/Solve.h"

using namespace open3d;
using namespace open3d::core;

// https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/lapacke_sgels_row.c.htm
int main() {
    std::vector<Device> devices{Device("CPU:0"), Device("CUDA:0")};
    std::vector<Dtype> dtypes{Dtype::Float32, Dtype::Float64};

    std::vector<float> A_vals{1.44,  -7.84, -4.39, 4.53,  -9.96, -0.28,
                              -3.24, 3.83,  -7.55, 3.24,  6.27,  -6.64,
                              8.34,  8.09,  5.28,  2.06,  7.08,  2.52,
                              0.74,  -2.47, -5.45, -5.70, -1.19, 4.70};
    std::vector<float> B_vals{8.58,  9.35,  8.26, -4.43, 8.48, -0.70,
                              -5.28, -0.26, 5.72, -7.36, 8.93, -2.52};
    // Solution:
    // -0.45 0.25
    // -0.85 -0.90
    // 0.71 0.63
    // 0.13 0.14

    Tensor A(A_vals, {6, 4}, core::Dtype::Float32, Device("CPU:0"));
    Tensor B(B_vals, {6, 2}, core::Dtype::Float32, Device("CPU:0"));

    std::cout << A.ToString() << "\n";
    std::cout << B.ToString() << "\n";

    for (auto dtype : dtypes) {
        for (auto device : devices) {
            Tensor A_device = A.Copy(device).To(dtype);
            Tensor B_device = B.Copy(device).To(dtype);
            Tensor X;
            Solve(A_device, B_device, X);
            std::cout << X.ToString() << "\n";
        }
    }
}
