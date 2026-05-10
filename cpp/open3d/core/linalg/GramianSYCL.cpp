// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <sycl/sycl.hpp>

#include "oneapi/mkl.hpp"
#include "open3d/core/SYCLContext.h"
#include "open3d/core/linalg/Gramian.h"
#include "open3d/core/linalg/LinalgUtils.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
void GramSYCL(void* A_data,
              void* B_data,
              int64_t m,
              int64_t n,
              Dtype dtype,
              const Device& device) {
    using namespace oneapi::mkl;
    sycl::queue queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
        scalar_t alpha = 1, beta = 0;
        blas::column_major::gemm(queue, transpose::N, transpose::T, n, n, m,
                                 alpha, static_cast<const scalar_t*>(A_data), n,
                                 static_cast<const scalar_t*>(A_data), n, beta,
                                 static_cast<scalar_t*>(B_data), n)
                .wait_and_throw();
    });

    void RowGramSYCL(void* A_data, void* B_data, int64_t m, int64_t n,
                     Dtype dtype, const Device& device) {
        using namespace oneapi::mkl;
        sycl::queue queue =
                sy::SYCLContext::GetInstance().GetDefaultQueue(device);
        DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
            scalar_t alpha = 1, beta = 0;
            blas::column_major::gemm(queue, transpose::T, transpose::N, m, m, n,
                                     alpha,
                                     static_cast<const scalar_t*>(A_data), n,
                                     static_cast<const scalar_t*>(A_data), n,
                                     beta, static_cast<scalar_t*>(B_data), m)
                    .wait_and_throw();
        });
    }

}  // namespace core
}  // namespace open3d
