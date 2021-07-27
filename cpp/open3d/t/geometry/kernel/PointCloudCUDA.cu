// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "open3d/t/geometry/kernel/PointCloudImpl.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace pointcloud {

void ProjectCUDA(
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

    // Pass 1: depth map
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
                if (!depth_indexer.InBoundary(u, v) || zc <= 0 ||
                    zc > depth_max) {
                    return;
                }

                float* depth_ptr = depth_indexer.GetDataPtr<float>(
                        static_cast<int64_t>(u), static_cast<int64_t>(v));
                float d = zc * depth_scale;
                float d_old = atomicExch(depth_ptr, d);
                if (d_old > 0) {
                    atomicMinf(depth_ptr, d_old);
                }
            });

    // Pass 2: color map
    if (!has_colors) return;

    NDArrayIndexer color_indexer(image_colors.value().get(), 2);
    float precision_bound = depth_scale * 1e-4;
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
                if (!depth_indexer.InBoundary(u, v) || zc <= 0 ||
                    zc > depth_max) {
                    return;
                }

                float dmap = *depth_indexer.GetDataPtr<float>(
                        static_cast<int64_t>(u), static_cast<int64_t>(v));
                float d = zc * depth_scale;
                if (d < dmap + precision_bound) {
                    uint8_t* color_ptr = color_indexer.GetDataPtr<uint8_t>(
                            static_cast<int64_t>(u), static_cast<int64_t>(v));
                    color_ptr[0] = static_cast<uint8_t>(
                            point_colors_ptr[3 * workload_idx + 0] * 255.0);
                    color_ptr[1] = static_cast<uint8_t>(
                            point_colors_ptr[3 * workload_idx + 1] * 255.0);
                    color_ptr[2] = static_cast<uint8_t>(
                            point_colors_ptr[3 * workload_idx + 2] * 255.0);
                }
            });
}

template <typename scalar_t>
__global__ void EstimateNormalsFromCovariancesCUDAKernel(
        const scalar_t* covariances_ptr,
        scalar_t* normals_ptr,
        const bool has_normals,
        const int64_t n) {
    const int64_t workload_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workload_idx >= n) return;

    int64_t covariances_offset = 9 * workload_idx;
    int64_t normals_offset = 3 * workload_idx;
    scalar_t normals_output[3] = {0};
    EstimatePointWiseNormalsWithFastEigen3x3<scalar_t>(
            covariances_ptr + covariances_offset, normals_output);

    if ((normals_output[0] * normals_output[0] +
         normals_output[1] * normals_output[1] +
         normals_output[2] * normals_output[2]) == 0.0 &&
        !has_normals) {
        normals_output[0] = 0.0;
        normals_output[1] = 0.0;
        normals_output[2] = 1.0;
    }
    if (has_normals) {
        if ((normals_ptr[normals_offset] * normals_output[0] +
             normals_ptr[normals_offset + 1] * normals_output[1] +
             normals_ptr[normals_offset + 2] * normals_output[2]) < 0.0) {
            normals_output[0] *= -1;
            normals_output[1] *= -1;
            normals_output[2] *= -1;
        }
    }

    normals_ptr[normals_offset] = normals_output[0];
    normals_ptr[normals_offset + 1] = normals_output[1];
    normals_ptr[normals_offset + 2] = normals_output[2];
}

void EstimateNormalsFromCovariancesCUDA(const core::Tensor& covariances,
                                        core::Tensor& normals,
                                        const bool has_normals) {
    core::Dtype dtype = covariances.GetDtype();
    int64_t n = covariances.GetLength();

    const dim3 blocks((n + 512 - 1) / 512);
    const dim3 threads(512);

    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
        EstimateNormalsFromCovariancesCUDAKernel<<<blocks, threads, 0,
                                                   core::cuda::GetStream()>>>(
                covariances.GetDataPtr<scalar_t>(),
                normals.GetDataPtr<scalar_t>(), has_normals, n);
    });
}

}  // namespace pointcloud
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
