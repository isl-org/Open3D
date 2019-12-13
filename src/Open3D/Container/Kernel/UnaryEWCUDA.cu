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

#include "Open3D/Container/CUDAState.cuh"
#include "Open3D/Container/CUDAUtils.h"
#include "Open3D/Container/Dispatch.h"
#include "Open3D/Container/Kernel/CUDALauncher.cuh"
#include "Open3D/Container/Tensor.h"

namespace open3d {
namespace kernel {

template <typename scalar_t>
static OPEN3D_HOST_DEVICE void CUDACopyElementKernel(const void* src,
                                                     void* dst) {
    *static_cast<scalar_t*>(dst) = *static_cast<const scalar_t*>(src);
}

void CopyCUDA(const Tensor& src, Tensor& dst) {
    // It has been checked that
    // - src and dst have the same shape, dtype
    // - at least one of src or dst is CUDA device
    SizeVector shape = src.GetShape();
    Dtype dtype = src.GetDtype();

    Device src_device = src.GetDevice();
    Device dst_device = dst.GetDevice();

    if (src_device.GetType() == Device::DeviceType::CUDA &&
        dst_device.GetType() == Device::DeviceType::CUDA) {
        if (src.IsContiguous() && dst.IsContiguous() &&
            src.GetShape() == dst.GetShape()) {
            // MemoryManager handles p2p and non-p2p device copy.
            MemoryManager::Memcpy(
                    dst.GetDataPtr(), dst_device, src.GetDataPtr(), src_device,
                    DtypeUtil::ByteSize(dtype) * shape.NumElements());
        } else if (src_device == dst_device) {
            // For more optimized version, one can check if P2P from src to dst
            // is enabled, then put synchronization with streams on both src and
            // dst to wait for copy kernel to complete.
            CUDASwitchDevice switcher(src_device);
            DISPATCH_DTYPE_TO_TEMPLATE(dtype, [&]() {
                Indexer indexer({src}, dst);
                CUDALauncher::LaunchUnaryEWKernel<scalar_t>(
                        indexer,
                        // Need to wrap as extended CUDA lamba function
                        [] OPEN3D_HOST_DEVICE(const void* src, void* dst) {
                            CUDACopyElementKernel<scalar_t>(src, dst);
                        });
            });
        } else {
            dst.CopyFrom(src.Clone(dst_device));
        }
    } else if (src_device.GetType() == Device::DeviceType::CPU &&
                       dst_device.GetType() == Device::DeviceType::CUDA ||
               src_device.GetType() == Device::DeviceType::CUDA &&
                       dst_device.GetType() == Device::DeviceType::CPU) {
        Tensor src_conti = src.Contiguous();  // No op if already contiguous
        if (dst.IsContiguous() && src.GetShape() == dst.GetShape()) {
            MemoryManager::Memcpy(
                    dst.GetDataPtr(), dst_device, src_conti.GetDataPtr(),
                    src_conti.GetDevice(),
                    DtypeUtil::ByteSize(dtype) * shape.NumElements());
        } else {
            dst.CopyFrom(src.Clone(dst_device));
        }
    } else {
        utility::LogError("Wrong device type {} -> {}", src_device.ToString(),
                          dst_device.ToString());
    }
}

}  // namespace kernel
}  // namespace open3d
