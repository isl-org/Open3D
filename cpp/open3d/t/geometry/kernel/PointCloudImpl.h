// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include <atomic>

#include "open3d/core/Dispatch.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/kernel/GeometryIndexer.h"
#include "open3d/t/geometry/kernel/GeometryMacros.h"
#include "open3d/t/geometry/kernel/PointCloud.h"
#include "open3d/utility/Console.h"

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
__device__ inline float atomicMinf(float* addr, float value) {
    float old;
    old = (value >= 0)
                  ? __int_as_float(atomicMin((int*)addr, __float_as_int(value)))
                  : __uint_as_float(atomicMax((unsigned int*)addr,
                                              __float_as_uint(value)));
    return old;
}
#endif

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace pointcloud {
#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
void UnprojectCUDA
#else
void UnprojectCPU
#endif
        (const core::Tensor& depth,
         const core::Tensor& color,
         core::Tensor& points,
         core::Tensor& point_colors,
         const core::Tensor& intrinsics,
         const core::Tensor& extrinsics,
         float depth_scale,
         float depth_max,
         int64_t stride) {

    NDArrayIndexer depth_indexer(depth, 2);
    NDArrayIndexer color_indexer(color, 2);

    bool process_color =
            color_indexer.GetShape(0) == depth_indexer.GetShape(0) &&
            color_indexer.GetShape(1) == depth_indexer.GetShape(1);

    TransformIndexer ti(intrinsics, extrinsics.Inverse(), 1.0f);

    // Output
    int64_t rows_strided = depth_indexer.GetShape(0) / stride;
    int64_t cols_strided = depth_indexer.GetShape(1) / stride;

    points = core::Tensor({rows_strided * cols_strided, 3},
                          core::Dtype::Float32, depth.GetDevice());
    if (process_color) {
        point_colors = core::Tensor({rows_strided * cols_strided, 3},
                                    core::Dtype::Float32, depth.GetDevice());
    } else {
        // Placeholder
        point_colors =
                core::Tensor({1, 3}, core::Dtype::Float32, depth.GetDevice());
    }
    NDArrayIndexer point_indexer(points, 1);
    NDArrayIndexer point_colors_indexer(point_colors, 1);

    // Counter
#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    core::Tensor count(std::vector<int>{0}, {}, core::Dtype::Int32,
                       depth.GetDevice());
    int* count_ptr = count.GetDataPtr<int>();
#else
    std::atomic<int> count_atomic(0);
    std::atomic<int>* count_ptr = &count_atomic;
#endif

    int64_t n = rows_strided * cols_strided;
#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    core::kernel::CUDALauncher::LaunchGeneralKernel(
            n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
#else
    core::kernel::CPULauncher::LaunchGeneralKernel(
            n, [&](int64_t workload_idx) {
#endif
                int64_t y = (workload_idx / cols_strided) * stride;
                int64_t x = (workload_idx % cols_strided) * stride;

                float d = *depth_indexer.GetDataPtrFromCoord<uint16_t>(x, y) /
                          depth_scale;
                if (d > 0 && d < depth_max) {
                    int idx = OPEN3D_ATOMIC_ADD(count_ptr, 1);

                    float x_c = 0, y_c = 0, z_c = 0;
                    ti.Unproject(static_cast<float>(x), static_cast<float>(y),
                                 d, &x_c, &y_c, &z_c);

                    float* vertex =
                            point_indexer.GetDataPtrFromCoord<float>(idx);
                    ti.RigidTransform(x_c, y_c, z_c, vertex + 0, vertex + 1,
                                      vertex + 2);

                    if (process_color) {
                        float* point_color =
                                point_colors_indexer.GetDataPtrFromCoord<float>(
                                        idx);
                        uint8_t* pixel_color =
                                color_indexer.GetDataPtrFromCoord<uint8_t>(x,
                                                                           y);
                        point_color[0] = pixel_color[0] / 255.0;
                        point_color[1] = pixel_color[1] / 255.0;
                        point_color[2] = pixel_color[2] / 255.0;
                    }
                }
            });
#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    int total_pts_count = count.Item<int>();
#else
    int total_pts_count = (*count_ptr).load();
#endif
    points = points.Slice(0, 0, total_pts_count);
    if (process_color) {
        point_colors = point_colors.Slice(0, 0, total_pts_count);
    }
}

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
void ProjectCUDA
#else
void ProjectCPU
#endif
        (core::Tensor& depth,
         core::Tensor& color,
         const core::Tensor& points,
         const core::Tensor& point_colors,
         const core::Tensor& intrinsics,
         const core::Tensor& extrinsics,
         float depth_scale,
         float depth_max) {
    int64_t n = points.GetLength();
    const float* points_ptr = static_cast<const float*>(points.GetDataPtr());
    const float* point_colors_ptr =
            static_cast<const float*>(point_colors.GetDataPtr());

    bool process_color = point_colors.GetLength() == points.GetLength();

    TransformIndexer transform_indexer(intrinsics, extrinsics, 1.0f);
    NDArrayIndexer depth_indexer(depth, 2);
    NDArrayIndexer color_indexer(color, 2);

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    core::kernel::CUDALauncher::LaunchGeneralKernel(
            n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
#else
    core::kernel::CPULauncher::LaunchGeneralKernel(
            n, [&](int64_t workload_idx) {
#endif
                float x = points_ptr[3 * workload_idx + 0];
                float y = points_ptr[3 * workload_idx + 1];
                float z = points_ptr[3 * workload_idx + 2];

                // coordinate in camera (in voxel -> in meter)
                float xc, yc, zc, u, v;
                transform_indexer.RigidTransform(x, y, z, &xc, &yc, &zc);

                // coordinate in image (in pixel)
                transform_indexer.Project(xc, yc, zc, &u, &v);
                if (!depth_indexer.InBoundary(u, v) || zc < 0 ||
                    zc > depth_max) {
                    return;
                }

                float* depth_ptr = depth_indexer.GetDataPtrFromCoord<float>(
                        static_cast<int64_t>(u), static_cast<int64_t>(v));
                float d = zc * depth_scale;
#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
                float d_old = atomicExch(depth_ptr, d);
                if (d_old > 0) {
                    float d_min = atomicMinf(depth_ptr, d_old);
                    if (process_color && d_min == d) {
                        uint8_t* color_ptr =
                                color_indexer.GetDataPtrFromCoord<uint8_t>(
                                        static_cast<int64_t>(u),
                                        static_cast<int64_t>(v));

                        color_ptr[0] = static_cast<uint8_t>(
                                point_colors_ptr[3 * workload_idx + 0] * 255.0);
                        color_ptr[1] = static_cast<uint8_t>(
                                point_colors_ptr[3 * workload_idx + 1] * 255.0);
                        color_ptr[2] = static_cast<uint8_t>(
                                point_colors_ptr[3 * workload_idx + 2] * 255.0);
                    }
                }

#else
#pragma omp critical
                {
                    if (*depth_ptr == 0 || *depth_ptr >= d) {
                        *depth_ptr = d;
                        if (process_color) {
                            uint8_t* color_ptr =
                                    color_indexer.GetDataPtrFromCoord<uint8_t>(
                                            static_cast<int64_t>(u),
                                            static_cast<int64_t>(v));

                            color_ptr[0] = static_cast<uint8_t>(
                                    point_colors_ptr[3 * workload_idx + 0] *
                                    255.0);
                            color_ptr[1] = static_cast<uint8_t>(
                                    point_colors_ptr[3 * workload_idx + 1] *
                                    255.0);
                            color_ptr[2] = static_cast<uint8_t>(
                                    point_colors_ptr[3 * workload_idx + 2] *
                                    255.0);
                        }
                    }
                }
#endif
            });
}  // namespace pointcloud
}  // namespace pointcloud
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
