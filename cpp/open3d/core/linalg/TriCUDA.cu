// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/Dispatch.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/linalg/TriImpl.h"

namespace open3d {
namespace core {

void TriuCUDA(const Tensor &A, Tensor &output, const int diagonal) {
    DISPATCH_DTYPE_TO_TEMPLATE(A.GetDtype(), [&]() {
        const scalar_t *A_ptr = static_cast<const scalar_t *>(A.GetDataPtr());
        scalar_t *output_ptr = static_cast<scalar_t *>(output.GetDataPtr());
        int cols = A.GetShape()[1];
        int n = A.GetShape()[0] * cols;

        core::ParallelFor(
                A.GetDevice(), n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
                    const int64_t idx = workload_idx / cols;
                    const int64_t idy = workload_idx % cols;
                    if (idy - idx >= diagonal) {
                        output_ptr[workload_idx] = A_ptr[idx * cols + idy];
                    }
                });
    });
}

void TrilCUDA(const Tensor &A, Tensor &output, const int diagonal) {
    DISPATCH_DTYPE_TO_TEMPLATE(A.GetDtype(), [&]() {
        const scalar_t *A_ptr = static_cast<const scalar_t *>(A.GetDataPtr());
        scalar_t *output_ptr = static_cast<scalar_t *>(output.GetDataPtr());
        int cols = A.GetShape()[1];
        int n = A.GetShape()[0] * cols;

        core::ParallelFor(
                A.GetDevice(), n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
                    const int64_t idx = workload_idx / cols;
                    const int64_t idy = workload_idx % cols;
                    if (idy - idx <= diagonal) {
                        output_ptr[workload_idx] = A_ptr[idx * cols + idy];
                    }
                });
    });
}

void TriulCUDA(const Tensor &A,
               Tensor &upper,
               Tensor &lower,
               const int diagonal) {
    DISPATCH_DTYPE_TO_TEMPLATE(A.GetDtype(), [&]() {
        const scalar_t *A_ptr = static_cast<const scalar_t *>(A.GetDataPtr());
        scalar_t *lower_ptr = static_cast<scalar_t *>(lower.GetDataPtr());
        scalar_t *upper_ptr = static_cast<scalar_t *>(upper.GetDataPtr());
        int cols = A.GetShape()[1];
        int n = A.GetShape()[0] * cols;

        core::ParallelFor(
                A.GetDevice(), n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
                    const int64_t idx = workload_idx / cols;
                    const int64_t idy = workload_idx % cols;
                    if (idy - idx < diagonal) {
                        lower_ptr[workload_idx] = A_ptr[idx * cols + idy];
                    } else if (idy - idx > diagonal) {
                        upper_ptr[workload_idx] = A_ptr[idx * cols + idy];
                    } else {
                        lower_ptr[workload_idx] = 1;
                        upper_ptr[workload_idx] = A_ptr[idx * cols + idy];
                    }
                });
    });
}

}  // namespace core
}  // namespace open3d
