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

#include "Open3D/Container/Kernel/UnaryEW.h"

#include "Open3D/Container/CudaUtils.cuh"
#include "Open3D/Container/Dispatch.h"
#include "Open3D/Container/Kernel/CUDALauncher.cuh"
#include "Open3D/Container/Tensor.h"

namespace open3d {
namespace kernel {

template <typename scalar_t>
void LaunchCopyKernel(const Tensor& src, Tensor& dst) {
    LaunchUnaryEWKernel<scalar_t>(
            src, dst,
            [] OPEN3D_HOST_DEVICE(const void* src, void* dst) -> void {
                *static_cast<scalar_t*>(dst) =
                        *static_cast<const scalar_t*>(src);
            });
}

void CopyCUDA(const Tensor& src, Tensor& dst) {
    // It has been checked that
    // - src and dst have the same shape, dtype, and dst
    // - dst must be contiguous
    // - at least one of src or dst is CUDA device
    SizeVector shape = src.GetShape();
    Dtype dtype = src.GetDtype();
    if (src.IsContiguous()) {
        MemoryManager::Memcpy(dst.GetDataPtr(), dst.GetDevice(),
                              src.GetDataPtr(), src.GetDevice(),
                              DtypeUtil::ByteSize(dtype) * shape.NumElements());
    } else {
        if (src.GetDevice() == dst.GetDevice()) {
            DISPATCH_DTYPE_TO_TEMPLATE(
                    dtype, [&]() { LaunchCopyKernel<scalar_t>(src, dst); });
        } else {
            // Works for both CPU -> GPU or GPU -> CPU
            Tensor src_conti = src.Copy(src.GetDevice());
            // Careful about the resursions
            CopyCUDA(src_conti, dst);
        }
    }
}

}  // namespace kernel
}  // namespace open3d
