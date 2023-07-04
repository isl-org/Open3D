// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/kernel/IndexReduction.h"

#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace kernel {

void IndexMean_(const Tensor& index, const Tensor& src, Tensor& dst) {
    if (dst.IsCPU()) {
        IndexMeanCPU_(index, src, dst);
    } else if (src.IsCUDA()) {
#ifdef BUILD_CUDA_MODULE
        IndexMeanCUDA_(index, src, dst);
#endif
    } else {
        utility::LogError("IndexMean_: Unimplemented device");
    }
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
