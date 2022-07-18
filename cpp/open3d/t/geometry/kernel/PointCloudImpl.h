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

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Dispatch.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/linalg/kernel/SVD3x3.h"
#include "open3d/core/nns/NearestNeighborSearch.h"
#include "open3d/t/geometry/Utility.h"
#include "open3d/t/geometry/kernel/GeometryIndexer.h"
#include "open3d/t/geometry/kernel/GeometryMacros.h"
#include "open3d/t/geometry/kernel/PointCloud.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace pointcloud {

#ifndef __CUDACC__
using std::abs;
using std::max;
using std::min;
using std::sqrt;
#endif

#if defined(__CUDACC__)
void UnprojectCUDA
#else
void UnprojectCPU
#endif
        (const core::Tensor& depth,
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

    points = core::Tensor({rows_strided * cols_strided, 3}, core::Float32,
                          depth.GetDevice());
    NDArrayIndexer point_indexer(points, 1);
    NDArrayIndexer colors_indexer;
    if (have_colors) {
        const auto& imcol = image_colors.value().get();
        image_colors_indexer = NDArrayIndexer{imcol, 2};
        colors.value().get() = core::Tensor({rows_strided * cols_strided, 3},
                                            core::Float32, imcol.GetDevice());
        colors_indexer = NDArrayIndexer(colors.value().get(), 1);
    }

    // Counter
#if defined(__CUDACC__)
    core::Tensor count(std::vector<int>{0}, {}, core::Int32, depth.GetDevice());
    int* count_ptr = count.GetDataPtr<int>();
#else
    std::atomic<int> count_atomic(0);
    std::atomic<int>* count_ptr = &count_atomic;
#endif

    int64_t n = rows_strided * cols_strided;

    DISPATCH_DTYPE_TO_TEMPLATE(depth.GetDtype(), [&]() {
        core::ParallelFor(
                depth.GetDevice(), n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
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
#if defined(__CUDACC__)
    int total_pts_count = count.Item<int>();
#else
    int total_pts_count = (*count_ptr).load();
#endif

#ifdef __CUDACC__
    core::cuda::Synchronize();
#endif
    points = points.Slice(0, 0, total_pts_count);
    if (have_colors) {
        colors.value().get() =
                colors.value().get().Slice(0, 0, total_pts_count);
    }
}

// This is a `two-pass` estimate method for covariance which is numerically more
// robust than the `textbook` method generally used for covariance computation.
template <typename scalar_t>
OPEN3D_HOST_DEVICE void EstimatePointWiseRobustNormalizedCovarianceKernel(
        const scalar_t* points_ptr,
        const int32_t* indices_ptr,
        const int32_t& indices_count,
        scalar_t* covariance_ptr) {
    if (indices_count < 3) {
        covariance_ptr[0] = 1.0;
        covariance_ptr[1] = 0.0;
        covariance_ptr[2] = 0.0;
        covariance_ptr[3] = 0.0;
        covariance_ptr[4] = 1.0;
        covariance_ptr[5] = 0.0;
        covariance_ptr[6] = 0.0;
        covariance_ptr[7] = 0.0;
        covariance_ptr[8] = 1.0;
        return;
    }

    double centroid[3] = {0};
    for (int32_t i = 0; i < indices_count; ++i) {
        int32_t idx = 3 * indices_ptr[i];
        centroid[0] += points_ptr[idx];
        centroid[1] += points_ptr[idx + 1];
        centroid[2] += points_ptr[idx + 2];
    }

    centroid[0] /= indices_count;
    centroid[1] /= indices_count;
    centroid[2] /= indices_count;

    // cumulants must always be Float64 to ensure precision.
    double cumulants[6] = {0};
    for (int32_t i = 0; i < indices_count; ++i) {
        int32_t idx = 3 * indices_ptr[i];
        const double x = static_cast<double>(points_ptr[idx]) - centroid[0];
        const double y = static_cast<double>(points_ptr[idx + 1]) - centroid[1];
        const double z = static_cast<double>(points_ptr[idx + 2]) - centroid[2];

        cumulants[0] += x * x;
        cumulants[1] += y * y;
        cumulants[2] += z * z;

        cumulants[3] += x * y;
        cumulants[4] += x * z;
        cumulants[5] += y * z;
    }

    // Using Bessel's correction (dividing by (n - 1) instead of n).
    // Refer: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    const double normalization_factor = static_cast<double>(indices_count - 1);
    for (int i = 0; i < 6; ++i) {
        cumulants[i] /= normalization_factor;
    }

    // Covariances(0, 0)
    covariance_ptr[0] = static_cast<scalar_t>(cumulants[0]);
    // Covariances(1, 1)
    covariance_ptr[4] = static_cast<scalar_t>(cumulants[1]);
    // Covariances(2, 2)
    covariance_ptr[8] = static_cast<scalar_t>(cumulants[2]);

    // Covariances(0, 1) = Covariances(1, 0)
    covariance_ptr[1] = static_cast<scalar_t>(cumulants[3]);
    covariance_ptr[3] = covariance_ptr[1];

    // Covariances(0, 2) = Covariances(2, 0)
    covariance_ptr[2] = static_cast<scalar_t>(cumulants[4]);
    covariance_ptr[6] = covariance_ptr[2];

    // Covariances(1, 2) = Covariances(2, 1)
    covariance_ptr[5] = static_cast<scalar_t>(cumulants[5]);
    covariance_ptr[7] = covariance_ptr[5];
}

#if defined(__CUDACC__)
void EstimateCovariancesUsingHybridSearchCUDA
#else
void EstimateCovariancesUsingHybridSearchCPU
#endif
        (const core::Tensor& points,
         core::Tensor& covariances,
         const double& radius,
         const int64_t& max_nn) {
    core::Dtype dtype = points.GetDtype();
    int64_t n = points.GetLength();

    core::nns::NearestNeighborSearch tree(points, core::Int32);
    bool check = tree.HybridIndex(radius);
    if (!check) {
        utility::LogError("Building FixedRadiusIndex failed.");
    }

    core::Tensor indices, distance, counts;
    std::tie(indices, distance, counts) =
            tree.HybridSearch(points, radius, max_nn);

    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
        const scalar_t* points_ptr = points.GetDataPtr<scalar_t>();
        int32_t* neighbour_indices_ptr = indices.GetDataPtr<int32_t>();
        int32_t* neighbour_counts_ptr = counts.GetDataPtr<int32_t>();
        scalar_t* covariances_ptr = covariances.GetDataPtr<scalar_t>();

        core::ParallelFor(
                points.GetDevice(), n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
                    // NNS [Hybrid Search].
                    const int32_t neighbour_offset = max_nn * workload_idx;
                    // Count of valid correspondences per point.
                    const int32_t neighbour_count =
                            neighbour_counts_ptr[workload_idx];
                    // Covariance is of shape {3, 3}, so it has an
                    // offset factor of 9 x workload_idx.
                    const int32_t covariances_offset = 9 * workload_idx;

                    EstimatePointWiseRobustNormalizedCovarianceKernel(
                            points_ptr,
                            neighbour_indices_ptr + neighbour_offset,
                            neighbour_count,
                            covariances_ptr + covariances_offset);
                });
    });

    core::cuda::Synchronize(points.GetDevice());
}

#if defined(__CUDACC__)
void EstimateCovariancesUsingKNNSearchCUDA
#else
void EstimateCovariancesUsingKNNSearchCPU
#endif
        (const core::Tensor& points,
         core::Tensor& covariances,
         const int64_t& max_nn) {
    core::Dtype dtype = points.GetDtype();
    int64_t n = points.GetLength();

    core::nns::NearestNeighborSearch tree(points, core::Int32);
    bool check = tree.KnnIndex();
    if (!check) {
        utility::LogError("Building KNN-Index failed.");
    }

    core::Tensor indices, distance;
    std::tie(indices, distance) = tree.KnnSearch(points, max_nn);

    indices = indices.Contiguous();
    int32_t nn_count = static_cast<int32_t>(indices.GetShape()[1]);

    if (nn_count < 3) {
        utility::LogError(
                "Not enough neighbors to compute Covariances / Normals. Try "
                "increasing the max_nn parameter.");
    }

    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
        auto points_ptr = points.GetDataPtr<scalar_t>();
        auto neighbour_indices_ptr = indices.GetDataPtr<int32_t>();
        auto covariances_ptr = covariances.GetDataPtr<scalar_t>();

        core::ParallelFor(
                points.GetDevice(), n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
                    // NNS [KNN Search].
                    const int32_t neighbour_offset = nn_count * workload_idx;
                    // Covariance is of shape {3, 3}, so it has an offset factor
                    // of 9 x workload_idx.
                    const int32_t covariances_offset = 9 * workload_idx;

                    EstimatePointWiseRobustNormalizedCovarianceKernel(
                            points_ptr,
                            neighbour_indices_ptr + neighbour_offset, nn_count,
                            covariances_ptr + covariances_offset);
                });
    });

    core::cuda::Synchronize(points.GetDevice());
}

template <typename scalar_t>
OPEN3D_HOST_DEVICE void ComputeEigenvector0(const scalar_t* A,
                                            const scalar_t eval0,
                                            scalar_t* eigen_vector0) {
    scalar_t row0[3] = {A[0] - eval0, A[1], A[2]};
    scalar_t row1[3] = {A[1], A[4] - eval0, A[5]};
    scalar_t row2[3] = {A[2], A[5], A[8] - eval0};

    scalar_t r0xr1[3], r0xr2[3], r1xr2[3];

    core::linalg::kernel::cross_3x1(row0, row1, r0xr1);
    core::linalg::kernel::cross_3x1(row0, row2, r0xr2);
    core::linalg::kernel::cross_3x1(row1, row2, r1xr2);

    scalar_t d0 = core::linalg::kernel::dot_3x1(r0xr1, r0xr1);
    scalar_t d1 = core::linalg::kernel::dot_3x1(r0xr2, r0xr2);
    scalar_t d2 = core::linalg::kernel::dot_3x1(r1xr2, r1xr2);

    scalar_t dmax = d0;
    int imax = 0;
    if (d1 > dmax) {
        dmax = d1;
        imax = 1;
    }
    if (d2 > dmax) {
        imax = 2;
    }

    if (imax == 0) {
        scalar_t sqrt_d = sqrt(d0);
        eigen_vector0[0] = r0xr1[0] / sqrt_d;
        eigen_vector0[1] = r0xr1[1] / sqrt_d;
        eigen_vector0[2] = r0xr1[2] / sqrt_d;
        return;
    } else if (imax == 1) {
        scalar_t sqrt_d = sqrt(d1);
        eigen_vector0[0] = r0xr2[0] / sqrt_d;
        eigen_vector0[1] = r0xr2[1] / sqrt_d;
        eigen_vector0[2] = r0xr2[2] / sqrt_d;
        return;
    } else {
        scalar_t sqrt_d = sqrt(d2);
        eigen_vector0[0] = r1xr2[0] / sqrt_d;
        eigen_vector0[1] = r1xr2[1] / sqrt_d;
        eigen_vector0[2] = r1xr2[2] / sqrt_d;
        return;
    }
}

template <typename scalar_t>
OPEN3D_HOST_DEVICE void ComputeEigenvector1(const scalar_t* A,
                                            const scalar_t* evec0,
                                            const scalar_t eval1,
                                            scalar_t* eigen_vector1) {
    scalar_t U[3];
    if (abs(evec0[0]) > abs(evec0[1])) {
        scalar_t inv_length =
                1.0 / sqrt(evec0[0] * evec0[0] + evec0[2] * evec0[2]);
        U[0] = -evec0[2] * inv_length;
        U[1] = 0.0;
        U[2] = evec0[0] * inv_length;
    } else {
        scalar_t inv_length =
                1.0 / sqrt(evec0[1] * evec0[1] + evec0[2] * evec0[2]);
        U[0] = 0.0;
        U[1] = evec0[2] * inv_length;
        U[2] = -evec0[1] * inv_length;
    }
    scalar_t V[3], AU[3], AV[3];
    core::linalg::kernel::cross_3x1(evec0, U, V);
    core::linalg::kernel::matmul3x3_3x1(A, U, AU);
    core::linalg::kernel::matmul3x3_3x1(A, V, AV);

    scalar_t m00 = core::linalg::kernel::dot_3x1(U, AU) - eval1;
    scalar_t m01 = core::linalg::kernel::dot_3x1(U, AV);
    scalar_t m11 = core::linalg::kernel::dot_3x1(V, AV) - eval1;

    scalar_t absM00 = abs(m00);
    scalar_t absM01 = abs(m01);
    scalar_t absM11 = abs(m11);
    scalar_t max_abs_comp;

    if (absM00 >= absM11) {
        max_abs_comp = max(absM00, absM01);
        if (max_abs_comp > 0) {
            if (absM00 >= absM01) {
                m01 /= m00;
                m00 = 1 / sqrt(1 + m01 * m01);
                m01 *= m00;
            } else {
                m00 /= m01;
                m01 = 1 / sqrt(1 + m00 * m00);
                m00 *= m01;
            }
            eigen_vector1[0] = m01 * U[0] - m00 * V[0];
            eigen_vector1[1] = m01 * U[1] - m00 * V[1];
            eigen_vector1[2] = m01 * U[2] - m00 * V[2];
            return;
        } else {
            eigen_vector1[0] = U[0];
            eigen_vector1[1] = U[1];
            eigen_vector1[2] = U[2];
            return;
        }
    } else {
        max_abs_comp = max(absM11, absM01);
        if (max_abs_comp > 0) {
            if (absM11 >= absM01) {
                m01 /= m11;
                m11 = 1 / sqrt(1 + m01 * m01);
                m01 *= m11;
            } else {
                m11 /= m01;
                m01 = 1 / sqrt(1 + m11 * m11);
                m11 *= m01;
            }
            eigen_vector1[0] = m11 * U[0] - m01 * V[0];
            eigen_vector1[1] = m11 * U[1] - m01 * V[1];
            eigen_vector1[2] = m11 * U[2] - m01 * V[2];
            return;
        } else {
            eigen_vector1[0] = U[0];
            eigen_vector1[1] = U[1];
            eigen_vector1[2] = U[2];
            return;
        }
    }
}

template <typename scalar_t>
OPEN3D_HOST_DEVICE void EstimatePointWiseNormalsWithFastEigen3x3(
        const scalar_t* covariance_ptr, scalar_t* normals_ptr) {
    // Based on:
    // https://www.geometrictools.com/Documentation/RobustEigenSymmetric3x3.pdf
    // which handles edge cases like points on a plane.
    scalar_t max_coeff = covariance_ptr[0];

    for (int i = 1; i < 9; ++i) {
        if (max_coeff < covariance_ptr[i]) {
            max_coeff = covariance_ptr[i];
        }
    }

    if (max_coeff == 0) {
        normals_ptr[0] = 0.0;
        normals_ptr[1] = 0.0;
        normals_ptr[2] = 0.0;
        return;
    }

    scalar_t A[9] = {0};

    for (int i = 0; i < 9; ++i) {
        A[i] = covariance_ptr[i] / max_coeff;
    }

    scalar_t norm = A[1] * A[1] + A[2] * A[2] + A[5] * A[5];

    if (norm > 0) {
        scalar_t eval[3];
        scalar_t evec0[3];
        scalar_t evec1[3];
        scalar_t evec2[3];

        scalar_t q = (A[0] + A[4] + A[8]) / 3.0;

        scalar_t b00 = A[0] - q;
        scalar_t b11 = A[4] - q;
        scalar_t b22 = A[8] - q;

        scalar_t p =
                sqrt((b00 * b00 + b11 * b11 + b22 * b22 + norm * 2.0) / 6.0);

        scalar_t c00 = b11 * b22 - A[5] * A[5];
        scalar_t c01 = A[1] * b22 - A[5] * A[2];
        scalar_t c02 = A[1] * A[5] - b11 * A[2];
        scalar_t det = (b00 * c00 - A[1] * c01 + A[2] * c02) / (p * p * p);

        scalar_t half_det = det * 0.5;
        half_det = min(max(half_det, static_cast<scalar_t>(-1.0)),
                       static_cast<scalar_t>(1.0));

        scalar_t angle = acos(half_det) / 3.0;
        const scalar_t two_thrids_pi = 2.09439510239319549;

        scalar_t beta2 = cos(angle) * 2.0;
        scalar_t beta0 = cos(angle + two_thrids_pi) * 2.0;
        scalar_t beta1 = -(beta0 + beta2);

        eval[0] = q + p * beta0;
        eval[1] = q + p * beta1;
        eval[2] = q + p * beta2;

        if (half_det >= 0) {
            ComputeEigenvector0<scalar_t>(A, eval[2], evec2);

            if (eval[2] < eval[0] && eval[2] < eval[1]) {
                normals_ptr[0] = evec2[0];
                normals_ptr[1] = evec2[1];
                normals_ptr[2] = evec2[2];

                return;
            }

            ComputeEigenvector1<scalar_t>(A, evec2, eval[1], evec1);

            if (eval[1] < eval[0] && eval[1] < eval[2]) {
                normals_ptr[0] = evec1[0];
                normals_ptr[1] = evec1[1];
                normals_ptr[2] = evec1[2];

                return;
            }

            normals_ptr[0] = evec1[1] * evec2[2] - evec1[2] * evec2[1];
            normals_ptr[1] = evec1[2] * evec2[0] - evec1[0] * evec2[2];
            normals_ptr[2] = evec1[0] * evec2[1] - evec1[1] * evec2[0];

            return;
        } else {
            ComputeEigenvector0<scalar_t>(A, eval[0], evec0);

            if (eval[0] < eval[1] && eval[0] < eval[2]) {
                normals_ptr[0] = evec0[0];
                normals_ptr[1] = evec0[1];
                normals_ptr[2] = evec0[2];
                return;
            }

            ComputeEigenvector1<scalar_t>(A, evec0, eval[1], evec1);

            if (eval[1] < eval[0] && eval[1] < eval[2]) {
                normals_ptr[0] = evec1[0];
                normals_ptr[1] = evec1[1];
                normals_ptr[2] = evec1[2];
                return;
            }

            normals_ptr[0] = evec0[1] * evec1[2] - evec0[2] * evec1[1];
            normals_ptr[1] = evec0[2] * evec1[0] - evec0[0] * evec1[2];
            normals_ptr[2] = evec0[0] * evec1[1] - evec0[1] * evec1[0];
            return;
        }
    } else {
        if (covariance_ptr[0] < covariance_ptr[4] &&
            covariance_ptr[0] < covariance_ptr[8]) {
            normals_ptr[0] = 1.0;
            normals_ptr[1] = 0.0;
            normals_ptr[2] = 0.0;
            return;
        } else if (covariance_ptr[0] < covariance_ptr[4] &&
                   covariance_ptr[0] < covariance_ptr[8]) {
            normals_ptr[0] = 0.0;
            normals_ptr[1] = 1.0;
            normals_ptr[2] = 0.0;
            return;
        } else {
            normals_ptr[0] = 0.0;
            normals_ptr[1] = 0.0;
            normals_ptr[2] = 1.0;
            return;
        }
    }
}

#if defined(__CUDACC__)
void EstimateNormalsFromCovariancesCUDA
#else
void EstimateNormalsFromCovariancesCPU
#endif
        (const core::Tensor& covariances,
         core::Tensor& normals,
         const bool has_normals) {
    core::Dtype dtype = covariances.GetDtype();
    int64_t n = covariances.GetLength();

    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
        const scalar_t* covariances_ptr = covariances.GetDataPtr<scalar_t>();
        scalar_t* normals_ptr = normals.GetDataPtr<scalar_t>();

        core::ParallelFor(
                covariances.GetDevice(), n,
                [=] OPEN3D_DEVICE(int64_t workload_idx) {
                    int32_t covariances_offset = 9 * workload_idx;
                    int32_t normals_offset = 3 * workload_idx;
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

    core::cuda::Synchronize(covariances.GetDevice());
}

template <typename scalar_t>
OPEN3D_HOST_DEVICE void EstimatePointWiseColorGradientKernel(
        const scalar_t* points_ptr,
        const scalar_t* normals_ptr,
        const scalar_t* colors_ptr,
        const int32_t& idx_offset,
        const int32_t* indices_ptr,
        const int32_t& indices_count,
        scalar_t* color_gradients_ptr) {
    if (indices_count < 4) {
        color_gradients_ptr[idx_offset] = 0;
        color_gradients_ptr[idx_offset + 1] = 0;
        color_gradients_ptr[idx_offset + 2] = 0;
    } else {
        scalar_t vt[3] = {points_ptr[idx_offset], points_ptr[idx_offset + 1],
                          points_ptr[idx_offset + 2]};

        scalar_t nt[3] = {normals_ptr[idx_offset], normals_ptr[idx_offset + 1],
                          normals_ptr[idx_offset + 2]};

        scalar_t it = (colors_ptr[idx_offset] + colors_ptr[idx_offset + 1] +
                       colors_ptr[idx_offset + 2]) /
                      3.0;

        scalar_t AtA[9] = {0};
        scalar_t Atb[3] = {0};

        // approximate image gradient of vt's tangential plane
        // projection (p') of a point p on a plane defined by
        // normal n, where o is the closest point to p on the
        // plane, is given by:
        // p' = p - [(p - o).dot(n)] * n p'
        // => p - [(p.dot(n) - s)] * n [where s = o.dot(n)]

        // Computing the scalar s.
        scalar_t s = vt[0] * nt[0] + vt[1] * nt[1] + vt[2] * nt[2];

        int i = 1;
        for (; i < indices_count; i++) {
            int64_t neighbour_idx_offset = 3 * indices_ptr[i];

            if (neighbour_idx_offset == -1) {
                break;
            }

            scalar_t vt_adj[3] = {points_ptr[neighbour_idx_offset],
                                  points_ptr[neighbour_idx_offset + 1],
                                  points_ptr[neighbour_idx_offset + 2]};

            // p' = p - d * n [where d = p.dot(n) - s]
            // Computing the scalar d.
            scalar_t d = vt_adj[0] * nt[0] + vt_adj[1] * nt[1] +
                         vt_adj[2] * nt[2] - s;

            // Computing the p' (projection of the point).
            scalar_t vt_proj[3] = {vt_adj[0] - d * nt[0], vt_adj[1] - d * nt[1],
                                   vt_adj[2] - d * nt[2]};

            scalar_t it_adj = (colors_ptr[neighbour_idx_offset + 0] +
                               colors_ptr[neighbour_idx_offset + 1] +
                               colors_ptr[neighbour_idx_offset + 2]) /
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
        scalar_t A[3] = {(i - 1) * nt[0], (i - 1) * nt[1], (i - 1) * nt[2]};

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

        core::linalg::kernel::solve_svd3x3(AtA, Atb,
                                           color_gradients_ptr + idx_offset);
    }
}

#if defined(__CUDACC__)
void EstimateColorGradientsUsingHybridSearchCUDA
#else
void EstimateColorGradientsUsingHybridSearchCPU
#endif
        (const core::Tensor& points,
         const core::Tensor& normals,
         const core::Tensor& colors,
         core::Tensor& color_gradients,
         const double& radius,
         const int64_t& max_nn) {
    core::Dtype dtype = points.GetDtype();
    int64_t n = points.GetLength();

    core::nns::NearestNeighborSearch tree(points, core::Int32);

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
        auto neighbour_indices_ptr = indices.GetDataPtr<int32_t>();
        auto neighbour_counts_ptr = counts.GetDataPtr<int32_t>();
        auto color_gradients_ptr = color_gradients.GetDataPtr<scalar_t>();

        core::ParallelFor(
                points.GetDevice(), n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
                    // NNS [Hybrid Search].
                    int32_t neighbour_offset = max_nn * workload_idx;
                    // Count of valid correspondences per point.
                    int32_t neighbour_count =
                            neighbour_counts_ptr[workload_idx];
                    int32_t idx_offset = 3 * workload_idx;

                    EstimatePointWiseColorGradientKernel(
                            points_ptr, normals_ptr, colors_ptr, idx_offset,
                            neighbour_indices_ptr + neighbour_offset,
                            neighbour_count, color_gradients_ptr);
                });
    });

    core::cuda::Synchronize(points.GetDevice());
}

#if defined(__CUDACC__)
void EstimateColorGradientsUsingKNNSearchCUDA
#else
void EstimateColorGradientsUsingKNNSearchCPU
#endif
        (const core::Tensor& points,
         const core::Tensor& normals,
         const core::Tensor& colors,
         core::Tensor& color_gradients,
         const int64_t& max_nn) {
    core::Dtype dtype = points.GetDtype();
    int64_t n = points.GetLength();

    core::nns::NearestNeighborSearch tree(points, core::Int32);

    bool check = tree.KnnIndex();
    if (!check) {
        utility::LogError("KnnIndex is not set.");
    }

    core::Tensor indices, distance;
    std::tie(indices, distance) = tree.KnnSearch(points, max_nn);

    indices = indices.To(core::Int32).Contiguous();
    int64_t nn_count = indices.GetShape()[1];

    if (nn_count < 4) {
        utility::LogError(
                "Not enough neighbors to compute Covariances / Normals. Try "
                "changing the search parameter.");
    }

    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
        auto points_ptr = points.GetDataPtr<scalar_t>();
        auto normals_ptr = normals.GetDataPtr<scalar_t>();
        auto colors_ptr = colors.GetDataPtr<scalar_t>();
        auto neighbour_indices_ptr = indices.GetDataPtr<int32_t>();
        auto color_gradients_ptr = color_gradients.GetDataPtr<scalar_t>();

        core::ParallelFor(
                points.GetDevice(), n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
                    int32_t neighbour_offset = max_nn * workload_idx;
                    int32_t idx_offset = 3 * workload_idx;

                    EstimatePointWiseColorGradientKernel(
                            points_ptr, normals_ptr, colors_ptr, idx_offset,
                            neighbour_indices_ptr + neighbour_offset, nn_count,
                            color_gradients_ptr);
                });
    });

    core::cuda::Synchronize(points.GetDevice());
}

}  // namespace pointcloud
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d