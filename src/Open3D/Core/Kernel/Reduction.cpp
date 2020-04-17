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
               bool keepdim,
               ReductionOpCode op_code) {
    // For ArgMin and ArgMax, keepdim == false, and dims can only contain one or
    // all dimensions.
    if (arg_reduce_ops.find(op_code) != arg_reduce_ops.end()) {
        if (keepdim) {
            utility::LogError("Arg-reduction keepdim must be false");
        }
        if (dims.size() != 1) {
            std::vector<bool> seen_dims(src.NumDims(), false);
            for (const int64_t& dim : dims) {
                seen_dims[dim] = true;
            }
            if (!std::all_of(seen_dims.begin(), seen_dims.end(),
                             [](bool seen) { return seen; })) {
                utility::LogError(
                        "Arg-reduction can only have 1 or all reduction "
                        "dimentions. However, dims = {}.",
                        dims);
            }
        }
    }

    SizeVector keepdim_shape =
            shape_util::ReductionShape(src.GetShape(), dims, true);
    SizeVector non_keepdim_shape =
            shape_util::ReductionShape(src.GetShape(), dims, false);
    if (keepdim && keepdim_shape != dst.GetShape()) {
        utility::LogError("Expected output shape {} but got {}.",
                          keepdim_shape.ToString(), dst.GetShape().ToString());
    }
    if (!keepdim && non_keepdim_shape != dst.GetShape()) {
        utility::LogError("Expected output shape {} but got {}.",
                          keepdim_shape.ToString(), dst.GetShape().ToString());
    }

    // Directly copy for non-reduction.
    if (dims.size() == 0) {
        dst.AsRvalue() = src;
        return;
    }

    // Always reshape to keepdim case. This reshaping is copy-free.
    if (!keepdim) {
        dst = dst.Reshape(keepdim_shape);
    }

    if (src.GetDevice() != dst.GetDevice()) {
        utility::LogError("Device mismatch {} != {}.",
                          src.GetDevice().ToString(),
                          dst.GetDevice().ToString());
    }

    Device::DeviceType device_type = src.GetDevice().GetType();
    if (device_type == Device::DeviceType::CPU) {
        ReductionCPU(src, dst, dims, keepdim, op_code);
    } else if (device_type == Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
        ReductionCUDA(src, dst, dims, keepdim, op_code);
#else
        utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
    } else {
        utility::LogError("Unimplemented device.");
    }

    if (!keepdim) {
        dst = dst.Reshape(non_keepdim_shape);
    }
}

}  // namespace kernel
}  // namespace open3d
