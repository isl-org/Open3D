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

#include "open3d/core/linalg/BlasWrapper.h"
#include "open3d/core/linalg/LinalgUtils.h"
#include "open3d/core/linalg/Matmul.h"
#include "open3d/utility/Console.h"

namespace open3d {
namespace core {

void MatmulCUDA(void* A_data,
                void* B_data,
                void* C_data,
                int64_t m,
                int64_t k,
                int64_t n,
                Dtype dtype) {
    cublasHandle_t handle = CuBLASContext::GetInstance()->GetHandle();
    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
        scalar_t alpha = 1, beta = 0;
        OPEN3D_CUBLAS_CHECK(
                gemm_cuda<scalar_t>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                                    &alpha,
                                    static_cast<const scalar_t*>(A_data), m,
                                    static_cast<const scalar_t*>(B_data), k,
                                    &beta, static_cast<scalar_t*>(C_data), m),
                "cuda gemm failed");
    });
}

}  // namespace core
}  // namespace open3d
