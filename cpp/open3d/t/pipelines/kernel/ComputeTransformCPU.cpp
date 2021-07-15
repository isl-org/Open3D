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

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <cmath>
#include <functional>
#include <vector>

#include "open3d/core/Dispatch.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/CPULauncher.h"
#include "open3d/t/pipelines/kernel/ComputeTransformImpl.h"
#include "open3d/t/pipelines/kernel/TransformationConverter.h"
#include "open3d/t/pipelines/registration/RobustKernel.h"
#include "open3d/t/pipelines/registration/RobustKernelImpl.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {

template <typename scalar_t, typename func_t>
static void ComputePosePointToPlaneKernelCPU(
        const scalar_t *source_points_ptr,
        const scalar_t *target_points_ptr,
        const scalar_t *target_normals_ptr,
        const int64_t *correspondence_indices,
        const int n,
        scalar_t *global_sum,
        func_t GetWeightFromRobustKernel) {
    // As, AtA is a symmetric matrix, we only need 21 elements instead of 36.
    // Atb is of shape {6,1}. Combining both, A_1x29 is a temp. storage
    // with [0:21] elements as AtA, [21:27] elements as Atb, 27th as residual
    // and 28th as inlier_count.
    std::vector<scalar_t> A_1x29(29, 0.0);

#ifdef _WIN32
    std::vector<scalar_t> zeros_29(29, 0.0);
    A_1x29 = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, n), zeros_29,
            [&](tbb::blocked_range<int> r, std::vector<scalar_t> A_reduction) {
                for (int workload_idx = r.begin(); workload_idx < r.end();
                     ++workload_idx) {
#else
    scalar_t *A_reduction = A_1x29.data();
#pragma omp parallel for reduction(+ : A_reduction[:29]) schedule(static)
    for (int workload_idx = 0; workload_idx < n; ++workload_idx) {
#endif
                    scalar_t J_ij[6];
                    scalar_t r = 0;

                    bool valid = kernel::GetJacobianPointToPlane<scalar_t>(
                            workload_idx, source_points_ptr, target_points_ptr,
                            target_normals_ptr, correspondence_indices, J_ij,
                            r);

                    scalar_t w = GetWeightFromRobustKernel(r);

                    if (valid) {
                        A_reduction[0] += J_ij[0] * w * J_ij[0];
                        A_reduction[1] += J_ij[1] * w * J_ij[0];
                        A_reduction[2] += J_ij[1] * w * J_ij[1];
                        A_reduction[3] += J_ij[2] * w * J_ij[0];
                        A_reduction[4] += J_ij[2] * w * J_ij[1];
                        A_reduction[5] += J_ij[2] * w * J_ij[2];
                        A_reduction[6] += J_ij[3] * w * J_ij[0];
                        A_reduction[7] += J_ij[3] * w * J_ij[1];
                        A_reduction[8] += J_ij[3] * w * J_ij[2];
                        A_reduction[9] += J_ij[3] * w * J_ij[3];
                        A_reduction[10] += J_ij[4] * w * J_ij[0];
                        A_reduction[11] += J_ij[4] * w * J_ij[1];
                        A_reduction[12] += J_ij[4] * w * J_ij[2];
                        A_reduction[13] += J_ij[4] * w * J_ij[3];
                        A_reduction[14] += J_ij[4] * w * J_ij[4];
                        A_reduction[15] += J_ij[5] * w * J_ij[0];
                        A_reduction[16] += J_ij[5] * w * J_ij[1];
                        A_reduction[17] += J_ij[5] * w * J_ij[2];
                        A_reduction[18] += J_ij[5] * w * J_ij[3];
                        A_reduction[19] += J_ij[5] * w * J_ij[4];
                        A_reduction[20] += J_ij[5] * w * J_ij[5];

                        A_reduction[21] += J_ij[0] * w * r;
                        A_reduction[22] += J_ij[1] * w * r;
                        A_reduction[23] += J_ij[2] * w * r;
                        A_reduction[24] += J_ij[3] * w * r;
                        A_reduction[25] += J_ij[4] * w * r;
                        A_reduction[26] += J_ij[5] * w * r;

                        A_reduction[27] += r * r;
                        A_reduction[28] += 1;
                    }
                }
#ifdef _WIN32
                return A_reduction;
            },
            // TBB: Defining reduction operation.
            [&](std::vector<scalar_t> a, std::vector<scalar_t> b) {
                std::vector<scalar_t> result(29);
                for (int j = 0; j < 29; ++j) {
                    result[j] = a[j] + b[j];
                }
                return result;
            });
#endif

#pragma omp parallel for schedule(static)
    for (int i = 0; i < 29; ++i) {
        global_sum[i] = A_1x29[i];
    }
}

void ComputePosePointToPlaneCPU(const core::Tensor &source_points,
                                const core::Tensor &target_points,
                                const core::Tensor &target_normals,
                                const core::Tensor &correspondence_indices,
                                core::Tensor &pose,
                                float &residual,
                                int &inlier_count,
                                const core::Dtype &dtype,
                                const core::Device &device,
                                const registration::RobustKernel &kernel) {
    int n = source_points.GetLength();

    core::Tensor global_sum = core::Tensor::Zeros({29}, dtype, device);

    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
        scalar_t *global_sum_ptr = global_sum.GetDataPtr<scalar_t>();

        DISPATCH_ROBUST_KERNEL_FUNCTION(
                kernel.type_, scalar_t, kernel.scaling_parameter_,
                kernel.shape_parameter_, [&]() {
                    kernel::ComputePosePointToPlaneKernelCPU(
                            source_points.GetDataPtr<scalar_t>(),
                            target_points.GetDataPtr<scalar_t>(),
                            target_normals.GetDataPtr<scalar_t>(),
                            correspondence_indices.GetDataPtr<int64_t>(), n,
                            global_sum_ptr, GetWeightFromRobustKernel);
                });
    });

    DecodeAndSolve6x6(global_sum, pose, residual, inlier_count);
}

template <typename scalar_t>
static void Get3x3SxyLinearSystem(const scalar_t *source_points_ptr,
                                  const scalar_t *target_points_ptr,
                                  const int64_t *correspondence_indices,
                                  const int &n,
                                  const core::Dtype &dtype,
                                  const core::Device &device,
                                  core::Tensor &Sxy,
                                  core::Tensor &target_mean,
                                  core::Tensor &source_mean,
                                  int &inlier_count) {
    // Calculating source_mean and target_mean, which are mean(x, y, z) of
    // source and target points respectively.
    std::vector<scalar_t> mean_1x7(7, 0.0);
    // Identity element for running_total reduction variable: zeros_6.
    std::vector<scalar_t> zeros_7(7, 0.0);

    mean_1x7 = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, n), zeros_7,
            [&](tbb::blocked_range<int> r,
                std::vector<scalar_t> mean_reduction) {
                for (int workload_idx = r.begin(); workload_idx < r.end();
                     ++workload_idx) {
                    if (correspondence_indices[workload_idx] != -1) {
                        int64_t target_idx =
                                3 * correspondence_indices[workload_idx];
                        mean_reduction[0] +=
                                source_points_ptr[3 * workload_idx];
                        mean_reduction[1] +=
                                source_points_ptr[3 * workload_idx + 1];
                        mean_reduction[2] +=
                                source_points_ptr[3 * workload_idx + 2];

                        mean_reduction[3] += target_points_ptr[target_idx];
                        mean_reduction[4] += target_points_ptr[target_idx + 1];
                        mean_reduction[5] += target_points_ptr[target_idx + 2];

                        mean_reduction[6] += 1;
                    }
                }
                return mean_reduction;
            },
            // TBB: Defining reduction operation.
            [&](std::vector<scalar_t> a, std::vector<scalar_t> b) {
                std::vector<scalar_t> result(7);
                for (int j = 0; j < 7; ++j) {
                    result[j] = a[j] + b[j];
                }
                return result;
            });

    for (int i = 0; i < 6; ++i) {
        mean_1x7[i] = mean_1x7[i] / mean_1x7[6];
    }

    // Calculating the Sxy for SVD.
    std::vector<scalar_t> sxy_1x9(9, 0.0);
    // Identity element for running total reduction variable: zeros_9.
    std::vector<scalar_t> zeros_9(9, 0.0);

    sxy_1x9 = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, n), zeros_9,
            [&](tbb::blocked_range<int> r,
                std::vector<scalar_t> sxy_1x9_reduction) {
                for (int workload_idx = r.begin(); workload_idx < r.end();
                     workload_idx++) {
                    if (correspondence_indices[workload_idx] != -1) {
                        for (int i = 0; i < 9; ++i) {
                            const int row = i % 3;
                            const int col = i / 3;
                            const int source_idx = 3 * workload_idx + row;
                            const int target_idx =
                                    3 * correspondence_indices[workload_idx] +
                                    col;
                            sxy_1x9_reduction[i] +=
                                    (source_points_ptr[source_idx] -
                                     mean_1x7[row]) *
                                    (target_points_ptr[target_idx] -
                                     mean_1x7[3 + col]);
                        }
                    }
                }
                return sxy_1x9_reduction;
            },
            // TBB: Defining reduction operation.
            [&](std::vector<scalar_t> a, std::vector<scalar_t> b) {
                std::vector<scalar_t> result(9);
                for (int j = 0; j < 9; ++j) {
                    result[j] = a[j] + b[j];
                }
                return result;
            });

    source_mean = core::Tensor::Empty({1, 3}, dtype, device);
    scalar_t *source_mean_ptr = source_mean.GetDataPtr<scalar_t>();

    target_mean = core::Tensor::Empty({1, 3}, dtype, device);
    scalar_t *target_mean_ptr = target_mean.GetDataPtr<scalar_t>();

    Sxy = core::Tensor::Empty({3, 3}, dtype, device);
    scalar_t *sxy_ptr = Sxy.GetDataPtr<scalar_t>();

    // Getting Tensor Sxy {3,3}, source_mean {3,1} and target_mean {3} from
    // temporary reduction variables. The shapes of source_mean and target_mean
    // are such, because it will be required in equation: t = source_mean -
    // R.Matmul(target_mean.T()).Reshape({-1}).
    for (int i = 0, j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; k++) {
            sxy_ptr[j * 3 + k] = sxy_1x9[i++] / mean_1x7[6];
        }
        source_mean_ptr[j] = mean_1x7[j];
        target_mean_ptr[j] = mean_1x7[j + 3];
    }

    inlier_count = static_cast<int64_t>(mean_1x7[6]);
}

void ComputeRtPointToPointCPU(const core::Tensor &source_points,
                              const core::Tensor &target_points,
                              const core::Tensor &corres,
                              core::Tensor &R,
                              core::Tensor &t,
                              int &inlier_count,
                              const core::Dtype &dtype,
                              const core::Device &device) {
    core::Tensor Sxy, target_mean, source_mean;

    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
        const scalar_t *source_points_ptr =
                source_points.GetDataPtr<scalar_t>();
        const scalar_t *target_points_ptr =
                target_points.GetDataPtr<scalar_t>();
        const int64_t *correspondence_indices = corres.GetDataPtr<int64_t>();

        int n = source_points.GetLength();

        Get3x3SxyLinearSystem(source_points_ptr, target_points_ptr,
                              correspondence_indices, n, dtype, device, Sxy,
                              target_mean, source_mean, inlier_count);
    });

    core::Tensor U, D, VT;
    std::tie(U, D, VT) = Sxy.SVD();
    core::Tensor S = core::Tensor::Eye(3, dtype, device);
    if (U.Det() * (VT.T()).Det() < 0) {
        S[-1][-1] = -1;
    }

    R = U.Matmul(S.Matmul(VT));
    t = (target_mean.Reshape({-1}) - R.Matmul(source_mean.T()).Reshape({-1}))
                .To(dtype);
}

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
