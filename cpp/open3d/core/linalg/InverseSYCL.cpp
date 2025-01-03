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
#include "open3d/core/linalg/Inverse.h"
#include "open3d/core/linalg/LinalgUtils.h"

namespace open3d {
namespace core {

void InverseSYCL(void* A_data,
                 void* ipiv_data,
                 void* output_data,
                 int64_t n,
                 Dtype dtype,
                 const Device& device) {
    using namespace oneapi::mkl;
    sycl::queue queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    int64_t lda = n;
    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
        // Use blob to ensure cleanup of scratchpad memory.
        int64_t scratchpad_size = std::max(
                lapack::getrf_scratchpad_size<scalar_t>(queue, n, n, lda),
                lapack::getri_scratchpad_size<scalar_t>(queue, n, lda));
        core::Blob scratchpad(scratchpad_size * sizeof(scalar_t), device);
        auto lu_done =
                lapack::getrf(queue, n, n, static_cast<scalar_t*>(A_data), lda,
                              static_cast<int64_t*>(ipiv_data),
                              static_cast<scalar_t*>(scratchpad.GetDataPtr()),
                              scratchpad_size);
        lapack::getri(queue, n, static_cast<scalar_t*>(A_data), lda,
                      static_cast<int64_t*>(ipiv_data),
                      static_cast<scalar_t*>(scratchpad.GetDataPtr()),
                      scratchpad_size, {lu_done})
                .wait_and_throw();
    });
}

}  // namespace core
}  // namespace open3d
