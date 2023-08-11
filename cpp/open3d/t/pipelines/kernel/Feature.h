// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Tensor.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {

void ComputeFPFHFeature(const core::Tensor &points,
                        const core::Tensor &normals,
                        const core::Tensor &indices,
                        const core::Tensor &distance2,
                        const core::Tensor &counts,
                        core::Tensor &fpfhs);

void ComputeFPFHFeatureCPU(const core::Tensor &points,
                           const core::Tensor &normals,
                           const core::Tensor &indices,
                           const core::Tensor &distance2,
                           const core::Tensor &counts,
                           core::Tensor &fpfhs);

#ifdef BUILD_CUDA_MODULE
void ComputeFPFHFeatureCUDA(const core::Tensor &points,
                            const core::Tensor &normals,
                            const core::Tensor &indices,
                            const core::Tensor &distance2,
                            const core::Tensor &counts,
                            core::Tensor &fpfhs);
#endif

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
