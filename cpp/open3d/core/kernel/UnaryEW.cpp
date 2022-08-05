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

#include "open3d/core/kernel/UnaryEW.h"

#include "open3d/core/ShapeUtil.h"
#include "open3d/core/Tensor.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace kernel {

void UnaryEW(const Tensor& src, Tensor& dst, const UnaryEWOpCode& op_code) {
    // Check shape
    if (!shape_util::CanBeBrocastedToShape(src.GetShape(), dst.GetShape())) {
        utility::LogError("Shape {} can not be broadcasted to {}.",
                          src.GetShape(), dst.GetShape());
    }

    // Dispatch to device
    Device src_device = src.GetDevice();
    Device dst_device = dst.GetDevice();
    if (src_device != dst_device) {
        utility::LogError("Source device {} != destination device {}.",
                          src_device.ToString(), dst_device.ToString());
    }

    if (src_device.IsCPU()) {
        UnaryEWCPU(src, dst, op_code);
    } else if (src_device.IsCUDA()) {
#ifdef BUILD_CUDA_MODULE
        UnaryEWCUDA(src, dst, op_code);
#else
        utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
    } else if (src_device.IsSYCL()) {
#ifdef BUILD_SYCL_MODULE
        UnaryEWSYCL(src, dst, op_code);
#else
        utility::LogError("Not compiled with SYCL, but SYCL device is used.");
#endif
    } else {
        utility::LogError("UnaryEW Unimplemented device");
    }
}

void Copy(const Tensor& src, Tensor& dst) {
    // Check shape.
    if (!shape_util::CanBeBrocastedToShape(src.GetShape(), dst.GetShape())) {
        utility::LogError("Shape {} can not be broadcasted to {}.",
                          src.GetShape(), dst.GetShape());
    }

    // Dispatch to device.
    if (src.IsCPU() && dst.IsCPU()) {
        CopyCPU(src, dst);
    } else if ((src.IsCUDA() && dst.IsCUDA()) ||
               (src.IsCPU() && dst.IsCUDA()) || (src.IsCUDA() && dst.IsCPU())) {
#ifdef BUILD_CUDA_MODULE
        CopyCUDA(src, dst);
#else
        utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
    } else if ((src.IsSYCL() && dst.IsSYCL()) ||
               (src.IsCPU() && dst.IsSYCL()) || (src.IsSYCL() && dst.IsCPU())) {
#ifdef BUILD_SYCL_MODULE
        CopySYCL(src, dst);
#else
        utility::LogError("Not compiled with SYCL, but SYCL device is used.");
#endif
    } else {
        utility::LogError("Unimplemented device {} -> {}.",
                          src.GetDevice().ToString(),
                          dst.GetDevice().ToString());
    }
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
