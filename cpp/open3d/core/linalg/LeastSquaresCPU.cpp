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

#include "open3d/core/linalg/LapackWrapper.h"
#include "open3d/core/linalg/LeastSquares.h"
#include "open3d/core/linalg/LinalgUtils.h"

namespace open3d {
namespace core {

void LeastSquaresCPU(void* A_data,
                     void* B_data,
                     int64_t m,
                     int64_t n,
                     int64_t k,
                     Dtype dtype,
                     const Device& device) {
    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
        OPEN3D_LAPACK_CHECK(
                gels_cpu<scalar_t>(LAPACK_COL_MAJOR, 'N', m, n, k,
                                   static_cast<scalar_t*>(A_data), m,
                                   static_cast<scalar_t*>(B_data),
                                   std::max(m, n)),
                "gels failed in LeastSquaresCPU");
    });
}

}  // namespace core
}  // namespace open3d
