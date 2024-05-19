// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cmath>
#include <cstring>

#include "open3d/core/Dtype.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/UnaryEW.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace kernel {

void CopySYCL(const Tensor& src, Tensor& dst) {
    // It has been checked that
    // - at least one of src or dst is SYCL device
    SizeVector shape = src.GetShape();
    Dtype src_dtype = src.GetDtype();
    Dtype dst_dtype = dst.GetDtype();
    Device dst_device = dst.GetDevice();
    Device src_device = src.GetDevice();

    if (src_dtype != dst_dtype) {
        utility::LogError(
                "CopySYCL: Dtype conversion from src to dst not implemented!");
    }
    if ((dst_device.IsSYCL() && !dst.IsContiguous()) ||
        (src_device.IsSYCL() && !src.IsContiguous())) {
        utility::LogError(
                "CopySYCL: NonContiguous SYCL tensor Copy not implemented!");
    }
    Tensor src_conti = src.Contiguous();  // No op if already contiguous
    if (dst.IsContiguous() && src.GetShape() == dst.GetShape() &&
        src_dtype == dst_dtype) {
        MemoryManager::Memcpy(dst.GetDataPtr(), dst_device,
                              src_conti.GetDataPtr(), src_conti.GetDevice(),
                              src_dtype.ByteSize() * shape.NumElements());
    } else {
        dst.CopyFrom(src_conti.To(dst_device));
    }
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
