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

// https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/sgesv_ex.c.htm
int main() {
    std::vector<Device> devices{Device("CPU:0"), Device("CUDA:0")};

    // Equation from https://www.mathworks.com/help/symbolic/linsolve.html
    std::vector<float> A_vals{2, 1, 1, -1, 1, -1, 1, 2, 3};
    std::vector<float> B_vals{2, 3, -10};
    Tensor A(A_vals, {3, 3}, core::Dtype::Float32, Device("CPU:0"));
    Tensor B(B_vals, {3, 1}, core::Dtype::Float32, Device("CPU:0"));

    std::cout << A.ToString() << "\n";
    std::cout << B.ToString() << "\n";

    for (auto device : devices) {
        Tensor A_device = A.Copy(device);
        Tensor B_device = B.Copy(device);
        Tensor X;
        Solve(A_device, B_device, X);
        std::cout << X.ToString() << "\n";
    }
}
