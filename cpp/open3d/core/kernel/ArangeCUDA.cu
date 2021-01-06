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

#include "open3d/core/Dispatch.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/Arange.h"
#include "open3d/core/kernel/CUDALauncher.cuh"

namespace open3d {
namespace core {
namespace kernel {

void ArangeCUDA(const Tensor& start,
                const Tensor& stop,
                const Tensor& step,
                Tensor& dst) {
    Dtype dtype = start.GetDtype();
    DISPATCH_DTYPE_TO_TEMPLATE(dtype, [&]() {
        scalar_t sstart = start.Item<scalar_t>();
        scalar_t sstep = step.Item<scalar_t>();
        scalar_t* dst_ptr = static_cast<scalar_t*>(dst.GetDataPtr());
        int64_t n = dst.GetLength();
        CUDALauncher::LaunchGeneralKernel(n, [=] OPEN3D_HOST_DEVICE(
                                                     int64_t workload_idx) {
            dst_ptr[workload_idx] =
                    sstart + static_cast<scalar_t>(sstep * workload_idx);
        });
    });
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
