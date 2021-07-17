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

#include "open3d/core/linalg/kernel/Matrix.h"
#include "open3d/t/geometry/kernel/GeometryMacros.h"
#include "open3d/t/geometry/kernel/PointCloud.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace pointcloud {

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
        int64_t stride);

void ProjectCPU(
        core::Tensor& depth,
        utility::optional<std::reference_wrapper<core::Tensor>> image_colors,
        const core::Tensor& points,
        utility::optional<std::reference_wrapper<const core::Tensor>> colors,
        const core::Tensor& intrinsics,
        const core::Tensor& extrinsics,
        float depth_scale,
        float depth_max);

#ifdef BUILD_CUDA_MODULE
void UnprojectCUDA(
        const core::Tensor& depth,
        utility::optional<std::reference_wrapper<const core::Tensor>>
                image_colors,
        core::Tensor& points,
        utility::optional<std::reference_wrapper<core::Tensor>> colors,
        const core::Tensor& intrinsics,
        const core::Tensor& extrinsics,
        float depth_scale,
        float depth_max,
        int64_t stride);

void ProjectCUDA(
        core::Tensor& depth,
        utility::optional<std::reference_wrapper<core::Tensor>> image_colors,
        const core::Tensor& points,
        utility::optional<std::reference_wrapper<const core::Tensor>> colors,
        const core::Tensor& intrinsics,
        const core::Tensor& extrinsics,
        float depth_scale,
        float depth_max);
#endif

void EstimateColorGradientsUsingHybridSearchCPU(const core::Tensor& points,
                                                const core::Tensor& normals,
                                                const core::Tensor& colors,
                                                core::Tensor& color_gradient,
                                                const double& radius,
                                                const int64_t& max_nn);

void EstimateCovariancesUsingHybridSearchCPU(const core::Tensor& points,
                                             core::Tensor& covariances,
                                             const double& radius,
                                             const int64_t& max_nn);

void EstimateCovariancesUsingKNNSearchCPU(const core::Tensor& points,
                                          core::Tensor& covariances,
                                          const int64_t& max_nn);

void EstimateNormalsFromCovariancesCPU(const core::Tensor& covariances,
                                       core::Tensor& normals,
                                       const bool has_normals);

#ifdef BUILD_CUDA_MODULE
void EstimateColorGradientsUsingHybridSearchCUDA(const core::Tensor& points,
                                                 const core::Tensor& normals,
                                                 const core::Tensor& colors,
                                                 core::Tensor& color_gradient,
                                                 const double& radius,
                                                 const int64_t& max_nn);

void EstimateCovariancesUsingHybridSearchCUDA(const core::Tensor& points,
                                              core::Tensor& covariances,
                                              const double& radius,
                                              const int64_t& max_nn);

void EstimateCovariancesUsingKNNSearchCUDA(const core::Tensor& points,
                                           core::Tensor& covariances,
                                           const int64_t& max_nn);

void EstimateNormalsFromCovariancesCUDA(const core::Tensor& covariances,
                                        core::Tensor& normals,
                                        const bool has_normals);
#endif

template <typename scalar_t>
OPEN3D_HOST_DEVICE void EstimatePointWiseCovarianceKernel(
        const scalar_t* points_ptr,
        const int64_t* indices_ptr,
        const int64_t& indices_count,
        scalar_t* covariance_ptr) {
    scalar_t cumulants[9] = {0};

    for (int64_t i = 0; i < indices_count; i++) {
        int64_t idx = 3 * indices_ptr[i];
        cumulants[0] += points_ptr[idx];
        cumulants[1] += points_ptr[idx + 1];
        cumulants[2] += points_ptr[idx + 2];
        cumulants[3] += points_ptr[idx] * points_ptr[idx];
        cumulants[4] += points_ptr[idx] * points_ptr[idx + 1];
        cumulants[5] += points_ptr[idx] * points_ptr[idx + 2];
        cumulants[6] += points_ptr[idx + 1] * points_ptr[idx + 1];
        cumulants[7] += points_ptr[idx + 1] * points_ptr[idx + 2];
        cumulants[8] += points_ptr[idx + 2] * points_ptr[idx + 2];
    }

    scalar_t num_indices = static_cast<scalar_t>(indices_count);
    cumulants[0] /= num_indices;
    cumulants[1] /= num_indices;
    cumulants[2] /= num_indices;
    cumulants[3] /= num_indices;
    cumulants[4] /= num_indices;
    cumulants[5] /= num_indices;
    cumulants[6] /= num_indices;
    cumulants[7] /= num_indices;
    cumulants[8] /= num_indices;

    // Covariances(0, 0)
    covariance_ptr[0] = cumulants[3] - cumulants[0] * cumulants[0];
    // Covariances(1, 1)
    covariance_ptr[4] = cumulants[6] - cumulants[1] * cumulants[1];
    // Covariances(2, 2)
    covariance_ptr[8] = cumulants[8] - cumulants[2] * cumulants[2];

    // Covariances(0, 1) = Covariances(1, 0)
    covariance_ptr[1] = cumulants[4] - cumulants[0] * cumulants[1];
    covariance_ptr[3] = covariance_ptr[1];

    // Covariances(0, 2) = Covariances(2, 0)
    covariance_ptr[2] = cumulants[5] - cumulants[0] * cumulants[2];
    covariance_ptr[6] = covariance_ptr[2];

    // Covariances(1, 2) = Covariances(2, 1)
    covariance_ptr[5] = cumulants[7] - cumulants[1] * cumulants[2];
    covariance_ptr[7] = covariance_ptr[5];

    return;
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
        max_abs_comp = OPEN3D_MAX(absM00, absM01);
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
        max_abs_comp = OPEN3D_MAX(absM11, absM01);
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

    for (int i = 1; i < 9; i++) {
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

    for (int i = 0; i < 9; i++) {
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
        half_det = OPEN3D_MIN(OPEN3D_MAX(half_det, -1.0), 1.0);

        scalar_t angle = acos(half_det) / 3.0;
        const scalar_t two_thrids_pi = 2.09439510239319549;

        scalar_t beta2 = cos(angle) * 2.0;
        scalar_t beta0 = cos(angle + two_thrids_pi) * 2.0;
        scalar_t beta1 = -(beta0 + beta2);

        eval[0] = q + p * beta0;
        eval[1] = q + p * beta1;
        eval[2] = q + p * beta2;

        if (half_det >= 0) {
            ComputeEigenvector0(A, eval[2], evec2);

            if (eval[2] < eval[0] && eval[2] < eval[1]) {
                normals_ptr[0] = evec2[0];
                normals_ptr[1] = evec2[1];
                normals_ptr[2] = evec2[2];

                return;
            }

            ComputeEigenvector1(A, evec2, eval[1], evec1);

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
            ComputeEigenvector0(A, eval[0], evec0);

            if (eval[0] < eval[1] && eval[0] < eval[2]) {
                normals_ptr[0] = evec0[0];
                normals_ptr[1] = evec0[1];
                normals_ptr[2] = evec0[2];
                return;
            }

            ComputeEigenvector1(A, evec0, eval[1], evec1);

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

}  // namespace pointcloud
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
