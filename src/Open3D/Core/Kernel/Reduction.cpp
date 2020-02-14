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

#include "Open3D/Core/Kernel/Reduction.h"
#include "Open3D/Core/SizeVector.h"

namespace open3d {
namespace kernel {

void Reduction(const Tensor& src,
               Tensor& dst,
               const SizeVector& dims,
               bool keep_dim,
               ReductionOpCode op_code) {
    SizeVector keep_dim_shape =
            shape_util::ReductionShape(src.GetShape(), dims, true);
    SizeVector non_keep_dim_shape =
            shape_util::ReductionShape(src.GetShape(), dims, false);
    if (keep_dim && keep_dim_shape != dst.GetShape()) {
        utility::LogError("Expected output shape {} but got {}.",
                          keep_dim_shape.ToString(), dst.GetShape().ToString());
    }
    if (!keep_dim && non_keep_dim_shape != dst.GetShape()) {
        utility::LogError("Expected output shape {} but got {}.",
                          keep_dim_shape.ToString(), dst.GetShape().ToString());
    }

    // Directly copy for non-reduction.
    if (dims.size() == 0) {
        dst.AsRvalue() = src;
        return;
    }

    // Always reshape to keep_dim case. This reshaping is copy-free.
    if (!keep_dim) {
        dst = dst.Reshape(keep_dim_shape);
    }

    if (src.GetDevice() != dst.GetDevice()) {
        utility::LogError("Device mismatch {} != {}.",
                          src.GetDevice().ToString(),
                          dst.GetDevice().ToString());
    }

    Device::DeviceType device_type = src.GetDevice().GetType();
    if (device_type == Device::DeviceType::CPU) {
        ReductionCPU(src, dst, dims, keep_dim, op_code);
    } else if (device_type == Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
        ReductionCUDA(src, dst, dims, keep_dim, op_code);
#else
        utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
    } else {
        utility::LogError("Unimplemented device.");
    }

    if (!keep_dim) {
        dst = dst.Reshape(non_keep_dim_shape);
    }
}

}  // namespace kernel
}  // namespace open3d
