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

#include "open3d/t/geometry/kernel/Image.h"

#include "open3d/core/CUDAUtils.h"
namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace image {

void To(const core::Tensor &src,
        core::Tensor &dst,
        double scale,
        double offset) {
    core::Device device = src.GetDevice();
    if (device.GetType() == core::Device::DeviceType::CPU) {
        ToCPU(src, dst, scale, offset);
    } else if (device.GetType() == core::Device::DeviceType::CUDA) {
        CUDA_CALL(ToCUDA, src, dst, scale, offset);
    } else {
        utility::LogError("Unimplemented device");
    }
}

void ClipTransform(const core::Tensor &src,
                   core::Tensor &dst,
                   float scale,
                   float min_value,
                   float max_value,
                   float clip_fill) {
    core::Device device = src.GetDevice();
    if (device.GetType() == core::Device::DeviceType::CPU) {
        ClipTransformCPU(src, dst, scale, min_value, max_value, clip_fill);
    } else if (device.GetType() == core::Device::DeviceType::CUDA) {
        CUDA_CALL(ClipTransformCUDA, src, dst, scale, min_value, max_value,
                  clip_fill);
    } else {
        utility::LogError("Unimplemented device");
    }
}

void PyrDownDepth(const core::Tensor &src,
                  core::Tensor &dst,
                  float diff_threshold,
                  float invalid_fill) {
    core::Device device = src.GetDevice();
    if (device.GetType() == core::Device::DeviceType::CPU) {
        PyrDownDepthCPU(src, dst, diff_threshold, invalid_fill);
    } else if (device.GetType() == core::Device::DeviceType::CUDA) {
        CUDA_CALL(PyrDownDepthCUDA, src, dst, diff_threshold, invalid_fill);
    } else {
        utility::LogError("Unimplemented device");
    }
}

void CreateVertexMap(const core::Tensor &src,
                     core::Tensor &dst,
                     const core::Tensor &intrinsics,
                     float invalid_fill) {
    core::Device device = src.GetDevice();
    static const core::Device host("CPU:0");

    core::Tensor intrinsics_d = intrinsics.To(host, core::Float64).Contiguous();
    if (device.GetType() == core::Device::DeviceType::CPU) {
        CreateVertexMapCPU(src, dst, intrinsics_d, invalid_fill);
    } else if (device.GetType() == core::Device::DeviceType::CUDA) {
        CUDA_CALL(CreateVertexMapCUDA, src, dst, intrinsics_d, invalid_fill);
    } else {
        utility::LogError("Unimplemented device");
    }
}

void CreateNormalMap(const core::Tensor &src,
                     core::Tensor &dst,
                     float invalid_fill) {
    core::Device device = src.GetDevice();
    if (device.GetType() == core::Device::DeviceType::CPU) {
        CreateNormalMapCPU(src, dst, invalid_fill);
    } else if (device.GetType() == core::Device::DeviceType::CUDA) {
        CUDA_CALL(CreateNormalMapCUDA, src, dst, invalid_fill);
    } else {
        utility::LogError("Unimplemented device");
    }
}

void ColorizeDepth(const core::Tensor &src,
                   core::Tensor &dst,
                   float scale,
                   float min_value,
                   float max_value) {
    core::Device device = src.GetDevice();
    if (device.GetType() == core::Device::DeviceType::CPU) {
        ColorizeDepthCPU(src, dst, scale, min_value, max_value);
    } else if (device.GetType() == core::Device::DeviceType::CUDA) {
        CUDA_CALL(ColorizeDepthCUDA, src, dst, scale, min_value, max_value);
    } else {
        utility::LogError("Unimplemented device");
    }
}

}  // namespace image
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
