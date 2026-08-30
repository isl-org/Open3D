// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/// \file PointCloudSYCL.cpp
/// \brief SYCL point-cloud kernels (see PointCloudImpl.h / PointCloud.h).

#include <cstdint>
#include <limits>
#include <sycl/sycl.hpp>

#include "open3d/core/ParallelFor.h"
#include "open3d/t/geometry/kernel/PointCloudImpl.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace pointcloud {

namespace {
constexpr uint64_t kEmptyPacked = std::numeric_limits<uint64_t>::max();
// Same packed-word atomicMin as ProjectCUDA (depth bits in high 32, point index
// in low 32).
OPEN3D_DEVICE void AtomicMinPackedUint64(uint64_t* addr, uint64_t val) {
    sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>(*addr)
            .fetch_min(val);
}

// CAS, not fetch_min: depth 0 means empty, so min(0, d) would never record the
// first hit.
OPEN3D_DEVICE void TryUpdateMinDepthSYCL(float* depth_ptr, float d) {
    auto depth_atomic_ref =
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(
                    *depth_ptr);
    float old = depth_atomic_ref.load(sycl::memory_order::relaxed);
    while (old == 0.0f || old > d) {
        if (depth_atomic_ref.compare_exchange_strong(
                    old, d, sycl::memory_order::relaxed,
                    sycl::memory_order::relaxed)) {
            break;
        }
    }
}

}  // namespace

// SYCL port of ProjectCUDA: same passes, atomics via sycl::atomic_ref.
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

    const int width = depth.GetShape(1);
    const int height = depth.GetShape(0);

    TransformIndexer transform_indexer(intrinsics, extrinsics, 1.0f);
    NDArrayIndexer depth_indexer(depth, 2);

    // Depth-only: see TryUpdateMinDepthSYCL; no winner buffer.
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
                    TryUpdateMinDepthSYCL(depth_ptr, d);
                });
        return;
    }

    core::Tensor packed_buffer = core::Tensor::Full(
            {height, width}, kEmptyPacked, core::UInt64, depth.GetDevice());
    uint64_t* packed_buffer_ptr = packed_buffer.GetDataPtr<uint64_t>();

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
                const uint32_t d_as_uint = sycl::bit_cast<uint32_t>(d);
                const uint64_t val = (static_cast<uint64_t>(d_as_uint) << 32) |
                                     static_cast<uint32_t>(workload_idx);

                AtomicMinPackedUint64(pixel_address, val);
            });

    NDArrayIndexer color_indexer(image_colors.value().get(), 2);
    // Pass 2: winning points unpack depth and write color from the packed pixel
    // word.
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
                *depth_ptr = sycl::bit_cast<float>(
                        static_cast<uint32_t>(packed >> 32));

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
