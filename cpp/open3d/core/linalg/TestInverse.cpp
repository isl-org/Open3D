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

#include "open3d/core/linalg/Inverse.h"

using namespace open3d;
using namespace open3d::core;

// https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/sgesv_ex.c.htm
int main() {
    std::vector<Device> devices{Device("CPU:0"), Device("CUDA:0")};
    std::vector<Dtype> dtypes{Dtype::Float32, Dtype::Float64};

    // Equation from https://www.mathworks.com/help/symbolic/linsolve.html
    std::vector<float> A_vals{0.8824, -0.1176, 0.1961, 0.1765, 0.1765,
                              0.0392, 0.0588,  0.0588, -0.0980};
    Tensor A(A_vals, {3, 3}, core::Dtype::Float32, Device("CPU:0"));

    std::cout << A.ToString() << "\n";

    for (auto dtype : dtypes) {
        for (auto device : devices) {
            Tensor A_device = A.Copy(device).To(dtype);
            Tensor A_inv;
            Inverse(A_device, A_inv);
            std::cout << A_inv.ToString() << "\n";
        }
    }
}
