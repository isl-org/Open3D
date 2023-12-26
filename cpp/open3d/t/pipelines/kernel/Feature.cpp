// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/pipelines/kernel/Feature.h"

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/TensorCheck.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {

void ComputeFPFHFeature(const core::Tensor &points,
                        const core::Tensor &normals,
                        const core::Tensor &indices,
                        const core::Tensor &distance2,
                        const core::Tensor &counts,
                        core::Tensor &fpfhs) {
    core::AssertTensorShape(fpfhs, {points.GetLength(), 33});
    const core::Tensor points_d = points.Contiguous();
    const core::Tensor normals_d = normals.Contiguous();
    const core::Tensor counts_d = counts.To(core::Int32);
    if (points_d.IsCPU()) {
        ComputeFPFHFeatureCPU(points_d, normals_d, indices, distance2, counts_d,
                              fpfhs);
    } else {
        core::CUDAScopedDevice scoped_device(points.GetDevice());
        CUDA_CALL(ComputeFPFHFeatureCUDA, points_d, normals_d, indices,
                  distance2, counts_d, fpfhs);
    }
}

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
