// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <sycl/sycl.hpp>

#include "open3d/core/ParallelFor.h"
#include "open3d/t/geometry/kernel/PointCloudImpl.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace pointcloud {

void ProjectSYCL(
        core::Tensor& depth,
        std::optional<std::reference_wrapper<core::Tensor>> image_colors,
        const core::Tensor& points,
        std::optional<std::reference_wrapper<const core::Tensor>> colors,
        const core::Tensor& intrinsics,
        const core::Tensor& extrinsics,
        float depth_scale,
        float depth_max) {
    const bool has_colors = image_colors.has_value();

    int64_t n = points.GetLength();

    const float* points_ptr = points.GetDataPtr<float>();
    const float* point_colors_ptr =
            has_colors ? colors.value().get().GetDataPtr<float>() : nullptr;

    core::Tensor depth_atomic = depth.Contiguous();
    TransformIndexer transform_indexer(intrinsics, extrinsics, 1.0f);
    NDArrayIndexer depth_indexer(depth_atomic, 2);

    core::ParallelFor(depth.GetDevice(), n, [=] OPEN3D_DEVICE(int64_t idx) {
        float x = points_ptr[3 * idx + 0];
        float y = points_ptr[3 * idx + 1];
        float z = points_ptr[3 * idx + 2];

        float xc, yc, zc, u, v;
        transform_indexer.RigidTransform(x, y, z, &xc, &yc, &zc);
        transform_indexer.Project(xc, yc, zc, &u, &v);
        u = round(u);
        v = round(v);
        if (!depth_indexer.InBoundary(u, v) || zc <= 0 || zc > depth_max) {
            return;
        }

        float* depth_ptr = depth_indexer.GetDataPtr<float>(
                static_cast<int64_t>(u), static_cast<int64_t>(v));
        float d = zc * depth_scale;
        auto depth_atomic_ref =
                sycl::atomic_ref<float, sycl::memory_order::acq_rel,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>(
                        *depth_ptr);
        float old = depth_atomic_ref.load();
        while (old == 0.0f || old > d) {
            if (depth_atomic_ref.compare_exchange_strong(old, d)) {
                break;
            }
        }
    });

    if (!has_colors) {
        depth = depth_atomic;
        return;
    }

    NDArrayIndexer color_indexer(image_colors.value().get(), 2);
    float precision_bound = depth_scale * 1e-4;
    core::Tensor depth_snapshot = depth_atomic;
    NDArrayIndexer depth_snapshot_indexer(depth_snapshot, 2);

    core::ParallelFor(depth.GetDevice(), n, [=] OPEN3D_DEVICE(int64_t idx) {
        float x = points_ptr[3 * idx + 0];
        float y = points_ptr[3 * idx + 1];
        float z = points_ptr[3 * idx + 2];

        float xc, yc, zc, u, v;
        transform_indexer.RigidTransform(x, y, z, &xc, &yc, &zc);
        transform_indexer.Project(xc, yc, zc, &u, &v);
        if (!depth_indexer.InBoundary(u, v) || zc <= 0 || zc > depth_max) {
            return;
        }

        float dmap = *depth_snapshot_indexer.GetDataPtr<float>(
                static_cast<int64_t>(u), static_cast<int64_t>(v));
        float d = zc * depth_scale;
        if (d < dmap + precision_bound) {
            float* color_ptr = color_indexer.GetDataPtr<float>(
                    static_cast<int64_t>(u), static_cast<int64_t>(v));
            color_ptr[0] = point_colors_ptr[3 * idx + 0];
            color_ptr[1] = point_colors_ptr[3 * idx + 1];
            color_ptr[2] = point_colors_ptr[3 * idx + 2];
        }
    });

    depth = depth_atomic;
}

}  // namespace pointcloud
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
