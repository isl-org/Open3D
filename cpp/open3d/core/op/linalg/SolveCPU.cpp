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

// https://
// software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/lapacke_sgesv_row.c.htm
#include <stdio.h>
#include <stdlib.h>

#include "open3d/core/op/linalg/Solve.h"

#include "lapack-netlib/LAPACKE/include/lapacke.h"

namespace open3d {
namespace core {

void SolveCPU(void* A_data,
              void* B_data,
              void* ipiv_data,
              void* X_data,  // unused place-holder for CPU
              int n,
              int m,
              Dtype dtype,
              const Device& device) {
    switch (dtype) {
        case Dtype::Float32: {
            // clang-format off
          LAPACKE_sgesv(LAPACK_ROW_MAJOR, n, m,
                        static_cast<float*>(A_data),
                        n,
                        static_cast<int*>(ipiv_data),
                        static_cast<float*>(B_data),
                        m);
            // clang-format on
            break;
        }

        case Dtype::Float64: {
            // clang-format off
          LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, m,
                        static_cast<double*>(A_data),
                        n,
                        static_cast<int*>(ipiv_data),
                        static_cast<double*>(B_data),
                        m);
            break;
            // clang-format on
        }

        default: {  // should never reach here
            utility::LogError("Unsupported dtype {} in CPU backend.",
                              DtypeUtil::ToString(dtype));
        }
    }
}

}  // namespace core
}  // namespace open3d
