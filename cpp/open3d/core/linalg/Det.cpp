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

#include "open3d/core/linalg/Det.h"

#include "open3d/core/Dispatch.h"
#include "open3d/core/linalg/LU.h"

namespace open3d {
namespace core {

double Det(const Tensor& A) {
    Tensor ipiv, output;
    LUIpiv(A, ipiv, output);
    // Sequential loop to compute determinant from LU output, is more efficient
    // on CPU.
    Tensor output_cpu = output.To(core::Device("CPU:0"));
    Tensor ipiv_cpu = ipiv.To(core::Device("CPU:0"));
    double det = 1.0;
    int n = A.GetShape()[0];

    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(A.GetDtype(), [&]() {
        scalar_t* output_ptr = output_cpu.GetDataPtr<scalar_t>();
        int* ipiv_ptr = static_cast<int*>(ipiv_cpu.GetDataPtr());

        for (int i = 0; i < n; i++) {
            det *= output_ptr[i * n + i];
            if (ipiv_ptr[i] != i) {
                det *= -1;
            }
        }
    });
    return det;
}

}  // namespace core
}  // namespace open3d
