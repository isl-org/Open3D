// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstdint>
#include <limits>

#include "open3d/t/geometry/kernel/PointCloudImpl.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace pointcloud {

namespace {
// Sentinel for pixels with no projected points (RGBD packed reduction).
constexpr uint64_t kEmptyPacked = std::numeric_limits<uint64_t>::max();
}  // namespace

// Parallel point splat: closest depth per pixel; RGBD picks one winning point
// for color.
void ProjectCUDA(
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

    const int width = depth.GetShape(1);
    const int height = depth.GetShape(0);

    TransformIndexer transform_indexer(intrinsics, extrinsics, 1.0f);
    NDArrayIndexer depth_indexer(depth, 2);

    // Depth-only: 0 means empty; exch+minf keeps the minimum positive depth per
    // pixel.
    if (!has_colors) {
        core::ParallelFor(
                depth.GetDevice(), n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
                    float x = points_ptr[3 * workload_idx + 0];
                    float y = points_ptr[3 * workload_idx + 1];
                    float z = points_ptr[3 * workload_idx + 2];

                    float xc, yc, zc, u, v;
                    transform_indexer.RigidTransform(x, y, z, &xc, &yc, &zc);
                    transform_indexer.Project(xc, yc, zc, &u, &v);
                    u = round(u);
                    v = round(v);
                    if (!depth_indexer.InBoundary(u, v) || zc <= 0 ||
                        zc > depth_max) {
                        return;
                    }

                    int64_t pu = static_cast<int64_t>(u),
                            pv = static_cast<int64_t>(v);
                    float* depth_ptr = depth_indexer.GetDataPtr<float>(pu, pv);
                    float d = zc * depth_scale;
                    float d_old = atomicExch(depth_ptr, d);
                    if (d_old > 0) {
                        atomicMinf(depth_ptr, d_old);
                    }
                });
        return;
    }

    core::Tensor packed_buffer = core::Tensor::Full(
            {height, width}, kEmptyPacked, core::UInt64, depth.GetDevice());
    uint64_t* packed_buffer_ptr = packed_buffer.GetDataPtr<uint64_t>();

    // Pass 1: 64-bit atomicMin on (depth, point_index) picks a deterministic
    // winner per pixel.
    core::ParallelFor(
            depth.GetDevice(), n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
                float x = points_ptr[3 * workload_idx + 0];
                float y = points_ptr[3 * workload_idx + 1];
                float z = points_ptr[3 * workload_idx + 2];

                float xc, yc, zc, u, v;
                transform_indexer.RigidTransform(x, y, z, &xc, &yc, &zc);
                transform_indexer.Project(xc, yc, zc, &u, &v);
                u = round(u);
                v = round(v);
                if (!depth_indexer.InBoundary(u, v) || zc <= 0 ||
                    zc > depth_max) {
                    return;
                }

                int64_t pu = static_cast<int64_t>(u),
                        pv = static_cast<int64_t>(v);
                float d = zc * depth_scale;

                uint64_t* pixel_address = packed_buffer_ptr + (pv * width + pu);
                uint32_t d_as_uint = __float_as_uint(d);
                // Lexicographic min: smaller depth wins; equal depth -> smaller
                // index.
                uint64_t val = (static_cast<uint64_t>(d_as_uint) << 32) |
                               (uint32_t)workload_idx;

                atomicMin(reinterpret_cast<unsigned long long*>(pixel_address),
                          static_cast<unsigned long long>(val));
            });

    NDArrayIndexer color_indexer(image_colors.value().get(), 2);
    // Pass 2: each winning point writes depth and color from the packed pixel
    // word.
    core::ParallelFor(
            depth.GetDevice(), n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
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

                if (!depth_indexer.InBoundary(u, v) || zc <= 0 ||
                    zc > depth_max) {
                    return;
                }

                int64_t pu = static_cast<int64_t>(u),
                        pv = static_cast<int64_t>(v);
                const uint64_t packed = packed_buffer_ptr[pv * width + pu];
                if (packed == kEmptyPacked) {
                    return;
                }

                const uint32_t winning_idx =
                        static_cast<uint32_t>(packed & 0xFFFFFFFF);
                if (winning_idx != static_cast<uint32_t>(workload_idx)) {
                    return;
                }

                float* depth_ptr = depth_indexer.GetDataPtr<float>(pu, pv);
                *depth_ptr =
                        __uint_as_float(static_cast<uint32_t>(packed >> 32));

                float* color_ptr = color_indexer.GetDataPtr<float>(pu, pv);
                color_ptr[0] = point_colors_ptr[3 * workload_idx + 0];
                color_ptr[1] = point_colors_ptr[3 * workload_idx + 1];
                color_ptr[2] = point_colors_ptr[3 * workload_idx + 2];
            });
}

}  // namespace pointcloud
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
