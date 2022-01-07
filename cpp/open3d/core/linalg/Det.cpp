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

#include "open3d/core/linalg/Det.h"

#include "open3d/core/linalg/LU.h"
#include "open3d/core/linalg/kernel/Matrix.h"

namespace open3d {
namespace core {

double Det(const Tensor& A) {
    AssertTensorDtypes(A, {Float32, Float64});
    const Dtype dtype = A.GetDtype();

    double det = 1.0;

    if (A.GetShape() == open3d::core::SizeVector({3, 3})) {
        DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
            core::Tensor A_3x3 =
                    A.To(core::Device("CPU:0"), false).Contiguous();
            const scalar_t* A_3x3_ptr = A_3x3.GetDataPtr<scalar_t>();
            det = static_cast<double>(linalg::kernel::det3x3(A_3x3_ptr));
        });
    } else if (A.GetShape() == open3d::core::SizeVector({2, 2})) {
        DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
            core::Tensor A_2x2 =
                    A.To(core::Device("CPU:0"), false).Contiguous();
            const scalar_t* A_2x2_ptr = A_2x2.GetDataPtr<scalar_t>();
            det = static_cast<double>(linalg::kernel::det2x2(A_2x2_ptr));
        });
    } else {
        Tensor ipiv, output;
        LUIpiv(A, ipiv, output);
        // Sequential loop to compute determinant from LU output, is more
        // efficient on CPU.
        Tensor output_cpu = output.To(core::Device("CPU:0"));
        Tensor ipiv_cpu = ipiv.To(core::Device("CPU:0"));
        int n = A.GetShape()[0];

        DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
            scalar_t* output_ptr = output_cpu.GetDataPtr<scalar_t>();
            int* ipiv_ptr = static_cast<int*>(ipiv_cpu.GetDataPtr());

            for (int i = 0; i < n; i++) {
                det *= output_ptr[i * n + i];
                if (ipiv_ptr[i] != i) {
                    det *= -1;
                }
            }
        });
    }
    return det;
}

}  // namespace core
}  // namespace open3d
