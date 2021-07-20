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

#include "open3d/core/kernel/CPULauncher.h"
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

    core::kernel::cpu_launcher::ParallelFor(n, [&](int64_t workload_idx) {
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

void EstimateColorGradientsUsingHybridSearchCPU(const core::Tensor& points,
                                                const core::Tensor& normals,
                                                const core::Tensor& colors,
                                                core::Tensor& color_gradients,
                                                const double& radius,
                                                const int64_t& max_nn) {
    core::Dtype dtype = points.GetDtype();
    int64_t n = points.GetLength();

    core::nns::NearestNeighborSearch tree(points);

    bool check = tree.HybridIndex(radius);
    if (!check) {
        utility::LogError(
                "NearestNeighborSearch::FixedRadiusIndex Index is not set.");
    }

    core::Tensor indices, distance, counts;
    std::tie(indices, distance, counts) =
            tree.HybridSearch(points, radius, max_nn);

    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
        auto points_ptr = points.GetDataPtr<scalar_t>();
        auto normals_ptr = normals.GetDataPtr<scalar_t>();
        auto colors_ptr = colors.GetDataPtr<scalar_t>();
        auto neighbour_indices_ptr = indices.GetDataPtr<int64_t>();
        auto neighbour_counts_ptr = counts.GetDataPtr<int64_t>();
        auto color_gradients_ptr = color_gradients.GetDataPtr<scalar_t>();

        core::kernel::cpu_launcher::ParallelFor(n, [&](int64_t workload_idx) {
            int64_t neighbour_offset = max_nn * workload_idx;
            int64_t neighbour_count = neighbour_counts_ptr[workload_idx];
            int64_t point_idx = 3 * workload_idx;

            if (neighbour_count >= 4) {
                scalar_t vt[3] = {points_ptr[point_idx],
                                  points_ptr[point_idx + 1],
                                  points_ptr[point_idx + 2]};

                scalar_t nt[3] = {normals_ptr[point_idx],
                                  normals_ptr[point_idx + 1],
                                  normals_ptr[point_idx + 2]};

                scalar_t it =
                        (colors_ptr[point_idx] + colors_ptr[point_idx + 1] +
                         colors_ptr[point_idx + 2]) /
                        3.0;

                scalar_t AtA[9] = {0};
                scalar_t Atb[3] = {0};

                // approximate image gradient of vt's tangential plane
                // projection (p') of a point p on a plane defined by
                // normal n, where o is the closest point to p on the
                // plane, is given by: p' = p - [(p - o).dot(n)] * n p'
                // = p - [(p.dot(n) - s)] * n [where s = o.dot(n)]
                // Computing the scalar s.
                scalar_t s = vt[0] * nt[0] + vt[1] * nt[1] + vt[2] * nt[2];

                int i = 1;
                for (i = 1; i < neighbour_count; i++) {
                    int64_t neighbour_idx =
                            3 * neighbour_indices_ptr[neighbour_offset + i];

                    if (neighbour_idx == -1) {
                        break;
                    }

                    scalar_t vt_adj[3] = {points_ptr[neighbour_idx],
                                          points_ptr[neighbour_idx + 1],
                                          points_ptr[neighbour_idx + 2]};

                    // p' = p - d * n [where d = p.dot(n) - s]
                    // Computing the scalar d.
                    scalar_t d = vt_adj[0] * nt[0] + vt_adj[1] * nt[1] +
                                 vt_adj[2] * nt[2] - s;

                    // Computing the p' (projection of the point).
                    scalar_t vt_proj[3] = {vt_adj[0] - d * nt[0],
                                           vt_adj[1] - d * nt[1],
                                           vt_adj[2] - d * nt[2]};

                    scalar_t it_adj = (colors_ptr[neighbour_idx + 0] +
                                       colors_ptr[neighbour_idx + 1] +
                                       colors_ptr[neighbour_idx + 2]) /
                                      3.0;

                    scalar_t A[3] = {vt_proj[0] - vt[0], vt_proj[1] - vt[1],
                                     vt_proj[2] - vt[2]};

                    AtA[0] += A[0] * A[0];
                    AtA[1] += A[1] * A[0];
                    AtA[2] += A[2] * A[0];
                    AtA[4] += A[1] * A[1];
                    AtA[5] += A[2] * A[1];
                    AtA[8] += A[2] * A[2];

                    scalar_t b = it_adj - it;

                    Atb[0] += A[0] * b;
                    Atb[1] += A[1] * b;
                    Atb[2] += A[2] * b;
                }

                // Orthogonal constraint.
                scalar_t A[3] = {(i - 1) * nt[0], (i - 1) * nt[1],
                                 (i - 1) * nt[2]};

                AtA[0] += A[0] * A[0];
                AtA[1] += A[0] * A[1];
                AtA[2] += A[0] * A[2];
                AtA[4] += A[1] * A[1];
                AtA[5] += A[1] * A[2];
                AtA[8] += A[2] * A[2];

                // Symmetry.
                AtA[3] = AtA[1];
                AtA[6] = AtA[2];
                AtA[7] = AtA[5];

                core::linalg::kernel::solve_svd3x3(
                        AtA, Atb, color_gradients_ptr + point_idx);

            } else {
                color_gradients_ptr[point_idx] = 0;
                color_gradients_ptr[point_idx + 1] = 0;
                color_gradients_ptr[point_idx + 2] = 0;
            }
        });
    });
}

void EstimateNormalsFromCovariancesCPU(const core::Tensor& covariances,
                                       core::Tensor& normals,
                                       const bool has_normals) {
    core::Dtype dtype = covariances.GetDtype();
    int64_t n = covariances.GetLength();

    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
        auto covariances_ptr = covariances.GetDataPtr<scalar_t>();
        auto normals_ptr = normals.GetDataPtr<scalar_t>();

        core::kernel::cpu_launcher::ParallelFor(n, [&](int64_t workload_idx) {
            int64_t covariances_offset = 9 * workload_idx;
            int64_t normals_offset = 3 * workload_idx;
            scalar_t normals_output[3] = {0};
            EstimatePointWiseNormalsWithFastEigen3x3(
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
                     normals_ptr[normals_offset + 2] * normals_output[2]) <
                    0.0) {
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
