// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/pipelines/kernel/Feature.h"

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/TensorCheck.h"

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
        const utility::optional<core::Tensor> &mask,
        const utility::optional<core::Tensor> &map_info_idx_to_point_idx) {
    if (mask.has_value()) {
        const int64_t size =
                mask.value().To(core::Int64).Sum({0}).Item<int64_t>();
        core::AssertTensorShape(fpfhs, {size, 33});
        core::AssertTensorShape(mask.value(), {points.GetLength()});
    } else {
        core::AssertTensorShape(fpfhs, {points.GetLength(), 33});
    }
    if (map_info_idx_to_point_idx.has_value()) {
        const bool is_radius_search = indices.GetShape().size() == 1;
        core::AssertTensorShape(
                map_info_idx_to_point_idx.value(),
                {counts.GetLength() - (is_radius_search ? 1 : 0)});
    }
    const core::Tensor points_d = points.Contiguous();
    const core::Tensor normals_d = normals.Contiguous();
    const core::Tensor counts_d = counts.To(core::Int32);
    if (points_d.IsCPU()) {
        ComputeFPFHFeatureCPU(points_d, normals_d, indices, distance2, counts_d,
                              fpfhs, mask, map_info_idx_to_point_idx);
    } else {
        core::CUDAScopedDevice scoped_device(points.GetDevice());
        CUDA_CALL(ComputeFPFHFeatureCUDA, points_d, normals_d, indices,
                  distance2, counts_d, fpfhs, mask, map_info_idx_to_point_idx);
    }
    utility::LogDebug(
            "[ComputeFPFHFeature] Computed {:d} features from "
            "input point cloud with {:d} points.",
            (int)fpfhs.GetLength(), (int)points.GetLength());
}

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
