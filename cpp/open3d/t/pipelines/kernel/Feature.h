// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Tensor.h"
#include "open3d/utility/Optional.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {

void ComputeFPFHFeature(
        const core::Tensor &points,
        const core::Tensor &normals,
        const core::Tensor &indices,
        const core::Tensor &distance2,
        const core::Tensor &counts,
        core::Tensor &fpfhs,
        const utility::optional<core::Tensor> &mask = utility::nullopt,
        const utility::optional<core::Tensor> &map_batch_info_idx_to_point_idx =
                utility::nullopt);

void ComputeFPFHFeatureCPU(
        const core::Tensor &points,
        const core::Tensor &normals,
        const core::Tensor &indices,
        const core::Tensor &distance2,
        const core::Tensor &counts,
        core::Tensor &fpfhs,
        const utility::optional<core::Tensor> &mask = utility::nullopt,
        const utility::optional<core::Tensor> &map_batch_info_idx_to_point_idx =
                utility::nullopt);

#ifdef BUILD_CUDA_MODULE
void ComputeFPFHFeatureCUDA(
        const core::Tensor &points,
        const core::Tensor &normals,
        const core::Tensor &indices,
        const core::Tensor &distance2,
        const core::Tensor &counts,
        core::Tensor &fpfhs,
        const utility::optional<core::Tensor> &mask = utility::nullopt,
        const utility::optional<core::Tensor> &map_batch_info_idx_to_point_idx =
                utility::nullopt);
#endif

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
