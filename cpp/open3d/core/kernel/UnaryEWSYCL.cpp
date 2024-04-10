// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
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
    SizeVector shape = src.GetShape();
    Dtype src_dtype = src.GetDtype();
    Dtype dst_dtype = dst.GetDtype();

    if (src_dtype != dst_dtype) {
        utility::LogError("CopySYCL: src and dst must have the same dtype");
    }

    if (src.GetShape() != dst.GetShape()) {
        utility::LogError("CopySYCL: src and dst must have the same shape");
    }

    if (src.GetShape().NumElements() == 0) {
        return;
    }

    MemoryManager::Memcpy(dst.GetDataPtr(), dst.GetDevice(), src.GetDataPtr(), src.GetDevice(),
                          src_dtype.ByteSize() * shape.NumElements());
}

} // namespace kernel
} // namespace core
} // namespace open3d
