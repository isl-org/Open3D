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

#include "open3d/core/algebra/Solver.h"

using namespace open3d;
using namespace open3d::core;

// https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/sgesv_ex.c.htm
int main() {
    std::vector<Device> devices{Device("CPU:0"), Device("CUDA:0")};

    std::vector<float> A_vals{6.80f,  -2.11f, 5.66f,  5.97f,  8.23f,
                              -6.05f, -3.30f, 5.36f,  -4.44f, 1.08f,
                              -0.45f, 2.58f,  -2.70f, 0.27f,  9.04f,
                              8.32f,  2.71f,  4.35f,  -7.17f, 2.14f,
                              -9.67f, -5.14f, -7.26f, 6.08f,  -6.87f};
    std::vector<float> B_vals{4.02f,  6.19f,  -8.22f, -7.57f, -3.03f,
                              -1.56f, 4.00f,  -8.67f, 1.75f,  2.86f,
                              9.81f,  -4.09f, -4.57f, -8.61f, 8.99f};
    Tensor A(A_vals, {5, 5}, core::Dtype::Float32, Device("CPU:0"));
    A = A.T();
    Tensor B(B_vals, {3, 5}, core::Dtype::Float32, Device("CPU:0"));
    B = B.T();

    std::cout << A.ToString() << "\n";
    std::cout << B.ToString() << "\n";

    for (auto device : devices) {
        Tensor A_device = A.Copy(device);
        Tensor B_device = B.Copy(device);
        Tensor X = Solve(A_device, B_device);
        std::cout << X.ToString() << "\n";
    }
}
