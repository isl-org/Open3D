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

#include <mkl.h>
#include "Matmul.h"

namespace open3d {
namespace core {
namespace detail {

// CPU converges to
// http://netlib.org/blas/#_level_3
// https://software.intel.com/content/www/us/en/develop/documentation/mkl-developer-reference-c/top/blas-and-sparse-blas-routines/blas-routines/blas-level-3-routines/cblas-gemm.html
// https://oneapi-src.github.io/oneDNN/group__dnnl__api__blas.html

void MatmulCPUBackend(Dtype dtype,
                      void* A_data,
                      void* B_data,
                      void* C_data,
                      int m,
                      int k,
                      int n) {
    switch (dtype) {
        case Dtype::Float32: {
            float alpha = 1, beta = 0;
            // clang-format off
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        m, n, k,  // dimensions
                        alpha,
                        static_cast<const float*>(A_data), m,
                        static_cast<const float*>(B_data), k,  // input and their leading dims
                        beta,
                        static_cast<float*>(C_data), m);  // output and its leading dim
            // clang-format on
            break;
        }

        case Dtype::Float64: {
            double alpha = 1, beta = 0;
            // clang-format off
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        m, n, k,  // dimensions
                        alpha,
                        static_cast<const double*>(A_data), m,
                        static_cast<const double*>(B_data), k,  // input and their leading dims
                        beta,
                        static_cast<double*>(C_data), m);  // output and its leading dim
            break;
            // clang-format on
        }

        default: {  // should never reach here
            utility::LogError("Unsupported dtype {} in CPU backend.",
                              DtypeUtil::ToString(dtype));
        }
    }
}

}  // namespace detail
}  // namespace core
}  // namespace open3d
