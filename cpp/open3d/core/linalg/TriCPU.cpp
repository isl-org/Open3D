// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "open3d/core/Dispatch.h"
#include "open3d/core/Indexer.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/linalg/TriImpl.h"

namespace open3d {
namespace core {

void TriuCPU(const Tensor &A, Tensor &output, const int diagonal) {
    DISPATCH_DTYPE_TO_TEMPLATE(A.GetDtype(), [&]() {
        const scalar_t *A_ptr = static_cast<const scalar_t *>(A.GetDataPtr());
        scalar_t *output_ptr = static_cast<scalar_t *>(output.GetDataPtr());
        int cols = A.GetShape()[1];
        int n = A.GetShape()[0] * cols;

        ParallelFor(A.GetDevice(), n, [&] OPEN3D_DEVICE(int64_t workload_idx) {
            const int64_t idx = workload_idx / cols;
            const int64_t idy = workload_idx % cols;
            if (idy - idx >= diagonal) {
                output_ptr[workload_idx] = A_ptr[idx * cols + idy];
            }
        });
    });
}

void TrilCPU(const Tensor &A, Tensor &output, const int diagonal) {
    DISPATCH_DTYPE_TO_TEMPLATE(A.GetDtype(), [&]() {
        const scalar_t *A_ptr = static_cast<const scalar_t *>(A.GetDataPtr());
        scalar_t *output_ptr = static_cast<scalar_t *>(output.GetDataPtr());
        int cols = A.GetShape()[1];
        int n = A.GetShape()[0] * cols;

        ParallelFor(A.GetDevice(), n, [&] OPEN3D_DEVICE(int64_t workload_idx) {
            const int64_t idx = workload_idx / cols;
            const int64_t idy = workload_idx % cols;
            if (idy - idx <= diagonal) {
                output_ptr[workload_idx] = A_ptr[idx * cols + idy];
            }
        });
    });
}

void TriulCPU(const Tensor &A,
              Tensor &upper,
              Tensor &lower,
              const int diagonal) {
    DISPATCH_DTYPE_TO_TEMPLATE(A.GetDtype(), [&]() {
        const scalar_t *A_ptr = static_cast<const scalar_t *>(A.GetDataPtr());
        scalar_t *upper_ptr = static_cast<scalar_t *>(upper.GetDataPtr());
        scalar_t *lower_ptr = static_cast<scalar_t *>(lower.GetDataPtr());
        int cols = A.GetShape()[1];
        int n = A.GetShape()[0] * cols;

        ParallelFor(A.GetDevice(), n, [&] OPEN3D_DEVICE(int64_t workload_idx) {
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
