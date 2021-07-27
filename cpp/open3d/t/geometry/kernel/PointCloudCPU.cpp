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
        if (!depth_indexer.InBoundary(u, v) || zc <= 0 || zc > depth_max) {
            return;
        }

        float* depth_ptr = depth_indexer.GetDataPtr<float>(
                static_cast<int64_t>(u), static_cast<int64_t>(v));
        float d = zc * depth_scale;
        // TODO: this can be wrong if ParallelFor is not implmented with OpenMP.
#pragma omp critical(ProjectCPU)
        {
            if (*depth_ptr == 0 || *depth_ptr >= d) {
                *depth_ptr = d;

                if (has_colors) {
                    uint8_t* color_ptr = color_indexer.GetDataPtr<uint8_t>(
                            static_cast<int64_t>(u), static_cast<int64_t>(v));

                    color_ptr[0] = static_cast<uint8_t>(
                            point_colors_ptr[3 * workload_idx + 0] * 255.0);
                    color_ptr[1] = static_cast<uint8_t>(
                            point_colors_ptr[3 * workload_idx + 1] * 255.0);
                    color_ptr[2] = static_cast<uint8_t>(
                            point_colors_ptr[3 * workload_idx + 2] * 255.0);
                }
            }
        }
    });
}

void EstimateNormalsFromCovariancesCPU(const core::Tensor& covariances,
                                       core::Tensor& normals,
                                       const bool has_normals) {
    core::Dtype dtype = covariances.GetDtype();
    int64_t n = covariances.GetLength();

    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
        const scalar_t* covariances_ptr = covariances.GetDataPtr<scalar_t>();
        scalar_t* normals_ptr = normals.GetDataPtr<scalar_t>();

        core::ParallelFor(
                covariances.GetDevice(), n, [&](int64_t workload_idx) {
                    int64_t covariances_offset = 9 * workload_idx;
                    int64_t normals_offset = 3 * workload_idx;
                    scalar_t normals_output[3] = {0};
                    EstimatePointWiseNormalsWithFastEigen3x3<scalar_t>(
                            covariances_ptr + covariances_offset,
                            normals_output);

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
                             normals_ptr[normals_offset + 1] *
                                     normals_output[1] +
                             normals_ptr[normals_offset + 2] *
                                     normals_output[2]) < 0.0) {
                            normals_output[0] *= -1;
                            normals_output[1] *= -1;
                            normals_output[2] *= -1;
                        }
                    }

                    normals_ptr[normals_offset] = normals_output[0];
                    normals_ptr[normals_offset + 1] = normals_output[1];
                    normals_ptr[normals_offset + 2] = normals_output[2];
                });
    });
}

}  // namespace pointcloud
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
