// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
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
    if (device.IsCPU()) {
        ToCPU(src, dst, scale, offset);
    } else if (device.IsCUDA()) {
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
    if (device.IsCPU()) {
        ClipTransformCPU(src, dst, scale, min_value, max_value, clip_fill);
    } else if (device.IsCUDA()) {
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
    if (device.IsCPU()) {
        PyrDownDepthCPU(src, dst, diff_threshold, invalid_fill);
    } else if (device.IsCUDA()) {
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
    if (device.IsCPU()) {
        CreateVertexMapCPU(src, dst, intrinsics_d, invalid_fill);
    } else if (device.IsCUDA()) {
        CUDA_CALL(CreateVertexMapCUDA, src, dst, intrinsics_d, invalid_fill);
    } else {
        utility::LogError("Unimplemented device");
    }
}

void CreateNormalMap(const core::Tensor &src,
                     core::Tensor &dst,
                     float invalid_fill) {
    core::Device device = src.GetDevice();
    if (device.IsCPU()) {
        CreateNormalMapCPU(src, dst, invalid_fill);
    } else if (device.IsCUDA()) {
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
    if (device.IsCPU()) {
        ColorizeDepthCPU(src, dst, scale, min_value, max_value);
    } else if (device.IsCUDA()) {
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
