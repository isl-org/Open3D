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
#include "open3d/core/linalg/LUImpl.h"
#include "open3d/core/linalg/LinalgUtils.h"

namespace open3d {
namespace core {

void LUSYCL(void* A_data,
            void* ipiv_data,
            int64_t m,
            int64_t n,
            Dtype dtype,
            const Device& device) {
    using namespace oneapi::mkl;
    sycl::queue queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    int64_t lda = m;
    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
        // Use blob to ensure cleanup of scratchpad memory.
        int64_t scratchpad_size =
                lapack::getrf_scratchpad_size<scalar_t>(queue, m, n, lda);
        core::Blob scratchpad(scratchpad_size * sizeof(scalar_t), device);
        lapack::getrf(queue, m, n, static_cast<scalar_t*>(A_data), lda,
                      static_cast<int64_t*>(ipiv_data),
                      static_cast<scalar_t*>(scratchpad.GetDataPtr()),
                      scratchpad_size)
                .wait_and_throw();
    });
}

}  // namespace core
}  // namespace open3d
