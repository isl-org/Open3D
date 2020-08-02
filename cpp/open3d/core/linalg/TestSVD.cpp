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

#include "open3d/core/linalg/SVD.h"

using namespace open3d;
using namespace open3d::core;

int main() {
    std::vector<Device> devices{Device("CPU:0"), Device("CUDA:0")};
    std::vector<Dtype> dtypes{Dtype::Float32, Dtype::Float64};

    std::vector<float> A_vals{
            8.79,  9.93,  9.83, 5.45,  3.16, 6.11, 6.91, 5.04,  -0.27, 7.98,
            -9.15, -7.93, 4.86, 4.85,  3.01, 9.57, 1.64, 8.83,  0.74,  5.80,
            -3.49, 4.02,  9.80, 10.00, 4.27, 9.84, 0.15, -8.99, -6.02, -5.31};
    Tensor A(A_vals, {6, 5}, core::Dtype::Float32, Device("CPU:0"));

    std::cout << A.ToString() << "\n";

    for (auto dtype : dtypes) {
        for (auto device : devices) {
            Tensor A_device = A.Copy(device).To(dtype);
            Tensor U, S, VT;
            SVD(A_device, U, S, VT);
            std::cout << U.ToString() << "\n";
            std::cout << S.ToString() << "\n";
            std::cout << VT.ToString() << "\n";
        }
    }
}
