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

#include <atomic>
#include <vector>

#include "open3d/core/Dispatch.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/CPULauncher.h"
#include "open3d/core/linalg/kernel/SVD3x3.h"
#include "open3d/core/nns/NearestNeighborSearch.h"
#include "open3d/t/geometry/Utility.h"
#include "open3d/t/geometry/kernel/GeometryIndexer.h"
#include "open3d/t/geometry/kernel/GeometryMacros.h"
#include "open3d/t/geometry/kernel/PointCloud.h"
#include "open3d/t/geometry/kernel/PointCloudImpl.h"
#include "open3d/utility/Logging.h"

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

void UnprojectCPU(
        const core::Tensor& depth,
        utility::optional<std::reference_wrapper<const core::Tensor>>
                image_colors,
        core::Tensor& points,
        utility::optional<std::reference_wrapper<core::Tensor>> colors,
        const core::Tensor& intrinsics,
        const core::Tensor& extrinsics,
        float depth_scale,
        float depth_max,
        int64_t stride) {
    const bool have_colors = image_colors.has_value();
    NDArrayIndexer depth_indexer(depth, 2);
    NDArrayIndexer image_colors_indexer;

    core::Tensor pose = t::geometry::InverseTransformation(extrinsics);
    TransformIndexer ti(intrinsics, pose, 1.0f);

    // Output
    int64_t rows_strided = depth_indexer.GetShape(0) / stride;
    int64_t cols_strided = depth_indexer.GetShape(1) / stride;

    points = core::Tensor({rows_strided * cols_strided, 3},
                          core::Dtype::Float32, depth.GetDevice());
    NDArrayIndexer point_indexer(points, 1);
    NDArrayIndexer colors_indexer;
    if (have_colors) {
        const auto& imcol = image_colors.value().get();
        image_colors_indexer = NDArrayIndexer{imcol, 2};
        colors.value().get() =
                core::Tensor({rows_strided * cols_strided, 3},
                             core::Dtype::Float32, imcol.GetDevice());
        colors_indexer = NDArrayIndexer(colors.value().get(), 1);
    }

    // Counter
    std::atomic<int> count_atomic(0);
    std::atomic<int>* count_ptr = &count_atomic;

    int64_t n = rows_strided * cols_strided;

    DISPATCH_DTYPE_TO_TEMPLATE(depth.GetDtype(), [&]() {
        core::kernel::cpu_launcher::ParallelFor(
                n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
                    int64_t y = (workload_idx / cols_strided) * stride;
                    int64_t x = (workload_idx % cols_strided) * stride;

                    float d = *depth_indexer.GetDataPtr<scalar_t>(x, y) /
                              depth_scale;
                    if (d > 0 && d < depth_max) {
                        int idx = OPEN3D_ATOMIC_ADD(count_ptr, 1);

                        float x_c = 0, y_c = 0, z_c = 0;
                        ti.Unproject(static_cast<float>(x),
                                     static_cast<float>(y), d, &x_c, &y_c,
                                     &z_c);

                        float* vertex = point_indexer.GetDataPtr<float>(idx);
                        ti.RigidTransform(x_c, y_c, z_c, vertex + 0, vertex + 1,
                                          vertex + 2);
                        if (have_colors) {
                            float* pcd_pixel =
                                    colors_indexer.GetDataPtr<float>(idx);
                            float* image_pixel =
                                    image_colors_indexer.GetDataPtr<float>(x,
                                                                           y);
                            *pcd_pixel = *image_pixel;
                            *(pcd_pixel + 1) = *(image_pixel + 1);
                            *(pcd_pixel + 2) = *(image_pixel + 2);
                        }
                    }
                });
    });
    int total_pts_count = (*count_ptr).load();

    points = points.Slice(0, 0, total_pts_count);
    if (have_colors) {
        colors.value().get() =
                colors.value().get().Slice(0, 0, total_pts_count);
    }
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

void EstimateCovariancesUsingHybridSearchCPU(const core::Tensor& points,
                                             core::Tensor& covariances,
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
        auto neighbour_indices_ptr = indices.GetDataPtr<int64_t>();
        auto neighbour_counts_ptr = counts.GetDataPtr<int64_t>();
        auto covariances_ptr = covariances.GetDataPtr<scalar_t>();

        core::kernel::cpu_launcher::ParallelFor(n, [&](int64_t workload_idx) {
            // NNS [Hybrid Search].
            int64_t neighbour_offset = max_nn * workload_idx;
            // Count of valid correspondences per point.
            int64_t neighbour_count = neighbour_counts_ptr[workload_idx];
            // Covariance is of shape {3, 3}, so it has an offset factor
            // of 9 x workload_idx.
            int64_t covariances_offset = 9 * workload_idx;

            if (neighbour_count >= 3) {
                EstimatePointWiseCovarianceKernel(
                        points_ptr, neighbour_indices_ptr + neighbour_offset,
                        neighbour_count, covariances_ptr + covariances_offset);
            } else {
                // Identity.
                covariances_ptr[covariances_offset] = 1.0;
                covariances_ptr[covariances_offset + 1] = 0.0;
                covariances_ptr[covariances_offset + 2] = 0.0;
                covariances_ptr[covariances_offset + 3] = 0.0;
                covariances_ptr[covariances_offset + 4] = 1.0;
                covariances_ptr[covariances_offset + 5] = 0.0;
                covariances_ptr[covariances_offset + 6] = 0.0;
                covariances_ptr[covariances_offset + 7] = 0.0;
                covariances_ptr[covariances_offset + 8] = 1.0;
            }
        });
    });
}

void EstimateCovariancesUsingKNNSearchCPU(const core::Tensor& points,
                                          core::Tensor& covariances,
                                          const int64_t& max_nn) {
    core::Dtype dtype = points.GetDtype();
    int64_t n = points.GetLength();

    core::nns::NearestNeighborSearch tree(points);

    bool check = tree.KnnIndex();
    if (!check) {
        utility::LogError("KnnIndex is not set.");
    }

    core::Tensor indices, distance;
    std::tie(indices, distance) = tree.KnnSearch(points, max_nn);

    indices = indices.Contiguous();
    int64_t nn_count = indices.GetShape()[1];

    if (nn_count < 3) {
        utility::LogError(
                " Not enought neighbors to compute Covariances / Normals. Try "
                "changing the search parameter.");
    }

    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
        auto points_ptr = points.GetDataPtr<scalar_t>();
        auto neighbour_indices_ptr = indices.GetDataPtr<int64_t>();
        auto covariances_ptr = covariances.GetDataPtr<scalar_t>();

        core::kernel::cpu_launcher::ParallelFor(n, [&](int64_t workload_idx) {
            // NNS [KNN Search].
            int64_t neighbour_offset = nn_count * workload_idx;
            // Covariance is of shape {3, 3}, so it has an offset factor
            // of 9 x workload_idx.
            int64_t covariances_offset = 9 * workload_idx;

            EstimatePointWiseCovarianceKernel(
                    points_ptr, neighbour_indices_ptr + neighbour_offset,
                    nn_count, covariances_ptr + covariances_offset);
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
