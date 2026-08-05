// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/kernel/UnaryEW.h"

#include "open3d/core/ShapeUtil.h"
#include "open3d/core/Tensor.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace kernel {

void UnaryEW(const Tensor& src, Tensor& dst, UnaryEWOpCode op_code) {
    // Check shape
    if (!shape_util::CanBeBrocastedToShape(src.GetShape(), dst.GetShape())) {
        utility::LogError("Shape {} can not be broadcasted to {}.",
                          src.GetShape(), dst.GetShape());
    }

    // Check dtype compatibility
    const auto float_only_ops = {UnaryEWOpCode::Sqrt,    UnaryEWOpCode::Sin,
                                 UnaryEWOpCode::Cos,     UnaryEWOpCode::Exp,
                                 UnaryEWOpCode::IsNan,   UnaryEWOpCode::IsInf,
                                 UnaryEWOpCode::IsFinite};
    Dtype src_dtype = src.GetDtype();
    if (std::find(float_only_ops.begin(), float_only_ops.end(), op_code) !=
                float_only_ops.end() &&
        src_dtype != core::Float32 && src_dtype != core::Float64) {
        utility::LogError("Only supports Float32 and Float64, but {} is used.",
                          src_dtype.ToString());
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
    } else if (src_device.IsSYCL()) {
#ifdef BUILD_SYCL_MODULE
        UnaryEWSYCL(src, dst, op_code);
#else
        utility::LogError("Not compiled with SYCL, but SYCL device is used.");
#endif
    } else if (src_device.IsCUDA()) {
#ifdef BUILD_CUDA_MODULE
        UnaryEWCUDA(src, dst, op_code);
#else
        utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
    } else {
        utility::LogError("UnaryEW Unimplemented device");
    }
}

void Copy(const Tensor& src, Tensor& dst) {
    // Check shape
    if (!shape_util::CanBeBrocastedToShape(src.GetShape(), dst.GetShape())) {
        utility::LogError("Shape {} can not be broadcasted to {}.",
                          src.GetShape(), dst.GetShape());
    }

    // Dispatch to device
    Device src_device = src.GetDevice();
    Device dst_device = dst.GetDevice();
    if ((!src_device.IsCPU() && !src_device.IsCUDA() && !src_device.IsSYCL()) ||
        (!dst_device.IsCPU() && !dst_device.IsCUDA() && !dst_device.IsSYCL())) {
        utility::LogError("Copy: Unimplemented device");
    }
    if (src_device.IsCPU() && dst_device.IsCPU()) {
        CopyCPU(src, dst);
    } else if ((src_device.IsCPU() || src_device.IsCUDA()) &&
               (dst_device.IsCPU() || dst_device.IsCUDA())) {
#ifdef BUILD_CUDA_MODULE
        CopyCUDA(src, dst);
#else
        utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
    } else if ((src_device.IsCPU() || src_device.IsSYCL()) &&
               (dst_device.IsCPU() || dst_device.IsSYCL())) {
#ifdef BUILD_SYCL_MODULE
        CopySYCL(src, dst);
#else
        utility::LogError("Not compiled with SYCL, but SYCL device is used.");
#endif
    } else {
        utility::LogError("Copy: SYCL <-> CUDA Unimplemented device");
    }
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
