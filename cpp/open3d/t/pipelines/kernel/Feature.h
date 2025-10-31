// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <optional>

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
                        core::Tensor &fpfhs,
                        const std::optional<core::Tensor> &mask = std::nullopt,
                        const std::optional<core::Tensor> &
                                map_batch_info_idx_to_point_idx = std::nullopt);

void ComputeFPFHFeatureCPU(
        const core::Tensor &points,
        const core::Tensor &normals,
        const core::Tensor &indices,
        const core::Tensor &distance2,
        const core::Tensor &counts,
        core::Tensor &fpfhs,
        const std::optional<core::Tensor> &mask = std::nullopt,
        const std::optional<core::Tensor> &map_batch_info_idx_to_point_idx =
                std::nullopt);

#ifdef BUILD_CUDA_MODULE
void ComputeFPFHFeatureCUDA(
        const core::Tensor &points,
        const core::Tensor &normals,
        const core::Tensor &indices,
        const core::Tensor &distance2,
        const core::Tensor &counts,
        core::Tensor &fpfhs,
        const std::optional<core::Tensor> &mask = std::nullopt,
        const std::optional<core::Tensor> &map_batch_info_idx_to_point_idx =
                std::nullopt);
#endif

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
