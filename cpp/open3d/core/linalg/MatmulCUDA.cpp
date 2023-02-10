// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/linalg/BlasWrapper.h"
#include "open3d/core/linalg/LinalgUtils.h"
#include "open3d/core/linalg/Matmul.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {

void MatmulCUDA(void* A_data,
                void* B_data,
                void* C_data,
                int64_t m,
                int64_t k,
                int64_t n,
                Dtype dtype,
                const Device& device) {
    cublasHandle_t handle = CuBLASContext::GetInstance().GetHandle(device);
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
