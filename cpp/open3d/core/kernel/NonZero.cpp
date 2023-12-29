// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/kernel/NonZero.h"

#include "open3d/core/Device.h"
#include "open3d/core/Tensor.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace kernel {

Tensor NonZero(const Tensor& src) {
    if (src.IsCPU()) {
        return NonZeroCPU(src);
    } else if (src.IsCUDA()) {
#ifdef BUILD_CUDA_MODULE
        return NonZeroCUDA(src);
#else
        utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
    } else {
        utility::LogError("NonZero: Unimplemented device");
    }
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
