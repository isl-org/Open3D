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

// https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/lapacke_sgesv_row.c.htm
#include <stdio.h>
#include <stdlib.h>

#include "open3d/core/linalg/LinalgUtils.h"
#include "open3d/core/linalg/Solve.h"

#include "open3d/core/linalg/LAPACK.h"

namespace open3d {
namespace core {

void SolveCPU(void* A_data,
              void* B_data,
              int m,
              int n,
              int k,
              Dtype dtype,
              const Device& device) {
    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
        gels_cpu<scalar_t>(LAPACK_COL_MAJOR, 'N', m, n, k,
                           static_cast<scalar_t*>(A_data), m,
                           static_cast<scalar_t*>(B_data), std::max(m, n));
    });
}

}  // namespace core
}  // namespace open3d
