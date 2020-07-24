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

#include "open3d/core/op/linalg/LinalgUtils.h"
#include "open3d/core/op/linalg/Matmul.h"
#include "open3d/utility/Console.h"
namespace open3d {
namespace core {

// Reference
// https://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemmbatched
// https://developer.nvidia.com/sites/default/files/akamai/cuda/files/Misc/mygpu.pdf

void MatmulCUDA(void* A_data,
                void* B_data,
                void* C_data,
                int m,
                int k,
                int n,
                Dtype dtype) {
    cublasHandle_t handle = CuBLASContext::GetInstance()->GetHandle();

    switch (dtype) {
        case Dtype::Float32: {
            float alpha = 1, beta = 0;
            OPEN3D_CUBLAS_CHECK(
                    cublasSgemm(handle, CUBLAS_OP_N,
                                CUBLAS_OP_N,  // A, B transpose flag
                                m, n, k,      // dimensions
                                &alpha, static_cast<const float*>(A_data), m,
                                static_cast<const float*>(B_data),
                                k,  // input and their leading dims
                                &beta, static_cast<float*>(C_data), m),
                    "cublasSgemm failed");
            break;
        }

        case Dtype::Float64: {
            double alpha = 1, beta = 0;
            OPEN3D_CUBLAS_CHECK(
                    cublasDgemm(handle, CUBLAS_OP_N,
                                CUBLAS_OP_N,  // A, B transpose flag
                                m, n, k,      // dimensions
                                &alpha, static_cast<const double*>(A_data), m,
                                static_cast<const double*>(B_data),
                                k,  // input and their leading dims
                                &beta, static_cast<double*>(C_data), m),
                    "cublasDgemm failed");
            break;
        }

        default: {  // should never reach here
            utility::LogError("Unsupported dtype {} in MatmulCUDA.",
                              DtypeUtil::ToString(dtype));
        }
    }
}

}  // namespace core
}  // namespace open3d
