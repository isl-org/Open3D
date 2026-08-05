// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <sycl/sycl.hpp>

#include "oneapi/mkl.hpp"
#include "open3d/core/Blob.h"
#include "open3d/core/SYCLContext.h"
#include "open3d/core/linalg/LinalgUtils.h"
#include "open3d/core/linalg/Solve.h"

namespace open3d {
namespace core {

void SolveSYCL(void* A_data,
               void* B_data,
               void* ipiv_data,
               int64_t n,
               int64_t k,
               Dtype dtype,
               const Device& device) {
    using namespace oneapi::mkl;
    sycl::queue queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    int64_t nrhs = k, lda = n, ldb = n;
    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
        int64_t scratchpad_size = lapack::gesv_scratchpad_size<scalar_t>(
                queue, n, nrhs, lda, ldb);
        // Use blob to ensure cleanup of scratchpad memory.
        Blob scratchpad(scratchpad_size * sizeof(scalar_t), device);
        lapack::gesv(queue, n, nrhs, static_cast<scalar_t*>(A_data), lda,
                     static_cast<int64_t*>(ipiv_data),
                     static_cast<scalar_t*>(B_data), ldb,
                     static_cast<scalar_t*>(scratchpad.GetDataPtr()),
                     scratchpad_size)
                .wait_and_throw();
    });
}

}  // namespace core
}  // namespace open3d
