// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/geometry/kernel/PointCloudImpl.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace pointcloud {

using std::round;

void ProjectCPU(
        core::Tensor& depth,
        utility::optional<std::reference_wrapper<core::Tensor>> image_colors,
        const core::Tensor& points,
        utility::optional<std::reference_wrapper<const core::Tensor>> colors,
        const core::Tensor& intrinsics,
        const core::Tensor& extrinsics,
        float depth_scale,
        float depth_max) {
    const bool has_colors = image_colors.has_value();

    int64_t n = points.GetLength();

    const float* points_ptr = points.GetDataPtr<float>();
    const float* point_colors_ptr =
            has_colors ? colors.value().get().GetDataPtr<float>() : nullptr;

    TransformIndexer transform_indexer(intrinsics, extrinsics, 1.0f);
    NDArrayIndexer depth_indexer(depth, 2);

    NDArrayIndexer color_indexer;
    if (has_colors) {
        color_indexer = NDArrayIndexer(image_colors.value().get(), 2);
    }

    core::ParallelFor(core::Device("CPU:0"), n, [&](int64_t workload_idx) {
        float x = points_ptr[3 * workload_idx + 0];
        float y = points_ptr[3 * workload_idx + 1];
        float z = points_ptr[3 * workload_idx + 2];

        // coordinate in camera (in voxel -> in meter)
        float xc, yc, zc, u, v;
        transform_indexer.RigidTransform(x, y, z, &xc, &yc, &zc);

        // coordinate in image (in pixel)
        transform_indexer.Project(xc, yc, zc, &u, &v);
        u = round(u);
        v = round(v);
        if (!depth_indexer.InBoundary(u, v) || zc <= 0 || zc > depth_max) {
            return;
        }

        float* depth_ptr = depth_indexer.GetDataPtr<float>(
                static_cast<int64_t>(u), static_cast<int64_t>(v));
        float d = zc * depth_scale;
        // TODO: this can be wrong if ParallelFor is not implemented with
        // OpenMP.
#pragma omp critical(ProjectCPU)
        {
            if (*depth_ptr == 0 || *depth_ptr >= d) {
                *depth_ptr = d;

                if (has_colors) {
                    float* color_ptr = color_indexer.GetDataPtr<float>(
                            static_cast<int64_t>(u), static_cast<int64_t>(v));

                    color_ptr[0] = point_colors_ptr[3 * workload_idx + 0];
                    color_ptr[1] = point_colors_ptr[3 * workload_idx + 1];
                    color_ptr[2] = point_colors_ptr[3 * workload_idx + 2];
                }
            }
        }
    });
}

}  // namespace pointcloud
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
