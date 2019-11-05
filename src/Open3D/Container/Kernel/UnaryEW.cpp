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

#include "Open3D/Container/Tensor.h"
#include "Open3D/Utility/Console.h"

namespace open3d {
namespace kernel {

void Copy(const Tensor& src, Tensor& dst) {
    // Check shape
    // TODO: in the future, we may support automatic broadcasting
    if (src.GetShape() != dst.GetShape()) {
        utility::LogError("src and dst tensor shape mismatch {} != {}",
                          src.GetShape().ToString(), dst.GetShape().ToString());
    }

    // Check strides
    // TODO: currently we require dst to be contiguous
    if (!dst.IsContiguous()) {
        utility::LogError("Unimplemented: dst must be contiguous");
    }

    // Check dtype
    // TODO: in the future, we may want to allow automatic casting
    if (src.GetDtype() != dst.GetDtype()) {
        utility::LogError("src and dst tensor dtype mismatch {} != {}",
                          DtypeUtil::ToString(src.GetDtype()),
                          DtypeUtil::ToString(dst.GetDtype()));
    }

    // Disbatch to device
    Device::DeviceType src_device_type = src.GetDevice().device_type_;
    Device::DeviceType dst_device_type = dst.GetDevice().device_type_;
    if ((src_device_type != Device::DeviceType::CPU &&
         src_device_type != Device::DeviceType::CUDA) ||
        (dst_device_type != Device::DeviceType::CPU &&
         dst_device_type != Device::DeviceType::CUDA)) {
        utility::LogError("Unimplemented device");
    }
    if (src_device_type == Device::DeviceType::CPU &&
        src_device_type == Device::DeviceType::CPU) {
        CopyCPU(src, dst);
    } else {
#ifdef BUILD_CUDA_MODULE
        CopyCUDA(src, dst);
#endif
    }
}

void IndexedGet(const Tensor& src,
                Tensor& dst,
                const std::vector<Tensor>& indices,
                const SizeVector& indexing_shape) {
    if (src.GetDevice().device_type_ == Device::DeviceType::CPU &&
        dst.GetDevice().device_type_ == Device::DeviceType::CPU) {
        utility::LogInfo("CPU");
        IndexedGetCPU(src, dst, indices, indexing_shape);

    } else if (src.GetDevice().device_type_ == Device::DeviceType::CUDA &&
               dst.GetDevice().device_type_ == Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
        utility::LogInfo("Cuda");
        IndexedGetCUDA(src, dst, indices, indexing_shape);
#endif
    }
}
void IndexedSet(const Tensor& src,
                Tensor& dst,
                const std::vector<Tensor>& indices) {}

}  // namespace kernel
}  // namespace open3d
