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
#include "open3d/core/ParallelFor.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/pipelines/kernel/RegistrationImpl.h"
#include "open3d/t/pipelines/kernel/TransformationConverter.h"
#include "open3d/t/pipelines/registration/RobustKernel.h"
#include "open3d/t/pipelines/registration/RobustKernelImpl.h"
#include "open3d/utility/Parallel.h"

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
#pragma omp parallel for reduction(+ : A_reduction[:29]) schedule(static) num_threads(utility::EstimateMaxThreads())
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
                        // Dump J, r into JtJ and Jtr
                        int i = 0;
                        for (int j = 0; j < 6; ++j) {
                            for (int k = 0; k <= j; ++k) {
                                A_reduction[i] += J_ij[j] * w * J_ij[k];
                                ++i;
                            }
                            A_reduction[21 + j] += J_ij[j] * w * r;
                        }
                        A_reduction[27] += r;
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

template <typename scalar_t, typename funct_t>
static void ComputePoseColoredICPKernelCPU(
        const scalar_t *source_points_ptr,
        const scalar_t *source_colors_ptr,
        const scalar_t *target_points_ptr,
        const scalar_t *target_normals_ptr,
        const scalar_t *target_colors_ptr,
        const scalar_t *target_color_gradients_ptr,
        const int64_t *correspondence_indices,
        const scalar_t &sqrt_lambda_geometric,
        const scalar_t &sqrt_lambda_photometric,
        const int n,
        scalar_t *global_sum,
        funct_t GetWeightFromRobustKernel) {
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
#pragma omp parallel for reduction(+ : A_reduction[:29]) schedule(static) num_threads(utility::EstimateMaxThreads())
    for (int workload_idx = 0; workload_idx < n; ++workload_idx) {
#endif
                    scalar_t J_G[6] = {0}, J_I[6] = {0};
                    scalar_t r_G = 0, r_I = 0;

                    bool valid = GetJacobianColoredICP<scalar_t>(
                            workload_idx, source_points_ptr, source_colors_ptr,
                            target_points_ptr, target_normals_ptr,
                            target_colors_ptr, target_color_gradients_ptr,
                            correspondence_indices, sqrt_lambda_geometric,
                            sqrt_lambda_photometric, J_G, J_I, r_G, r_I);

                    scalar_t w_G = GetWeightFromRobustKernel(r_G);
                    scalar_t w_I = GetWeightFromRobustKernel(r_I);

                    if (valid) {
                        // Dump J, r into JtJ and Jtr
                        int i = 0;
                        for (int j = 0; j < 6; ++j) {
                            for (int k = 0; k <= j; ++k) {
                                A_reduction[i] += J_G[j] * w_G * J_G[k] +
                                                  J_I[j] * w_I * J_I[k];
                                ++i;
                            }
                            A_reduction[21 + j] +=
                                    J_G[j] * w_G * r_G + J_I[j] * w_I * r_I;
                        }
                        A_reduction[27] += r_G * r_G + r_I * r_I;
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

    for (int i = 0; i < 29; ++i) {
        global_sum[i] = A_1x29[i];
    }
}

void ComputePoseColoredICPCPU(const core::Tensor &source_points,
                              const core::Tensor &source_colors,
                              const core::Tensor &target_points,
                              const core::Tensor &target_normals,
                              const core::Tensor &target_colors,
                              const core::Tensor &target_color_gradients,
                              const core::Tensor &correspondence_indices,
                              core::Tensor &pose,
                              float &residual,
                              int &inlier_count,
                              const core::Dtype &dtype,
                              const core::Device &device,
                              const registration::RobustKernel &kernel,
                              const double &lambda_geometric) {
    int n = source_points.GetLength();

    core::Tensor global_sum = core::Tensor::Zeros({29}, dtype, device);

    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
        scalar_t sqrt_lambda_geometric =
                static_cast<scalar_t>(sqrt(lambda_geometric));
        scalar_t sqrt_lambda_photometric =
                static_cast<scalar_t>(sqrt(1.0 - lambda_geometric));
        DISPATCH_ROBUST_KERNEL_FUNCTION(
                kernel.type_, scalar_t, kernel.scaling_parameter_,
                kernel.shape_parameter_, [&]() {
                    kernel::ComputePoseColoredICPKernelCPU(
                            source_points.GetDataPtr<scalar_t>(),
                            source_colors.GetDataPtr<scalar_t>(),
                            target_points.GetDataPtr<scalar_t>(),
                            target_normals.GetDataPtr<scalar_t>(),
                            target_colors.GetDataPtr<scalar_t>(),
                            target_color_gradients.GetDataPtr<scalar_t>(),
                            correspondence_indices.GetDataPtr<int64_t>(),
                            sqrt_lambda_geometric, sqrt_lambda_photometric, n,
                            global_sum.GetDataPtr<scalar_t>(),
                            GetWeightFromRobustKernel);
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
    // mean_1x7[6] is the number of total valid correspondences.
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

    if (mean_1x7[6] == 0) {
        utility::LogError("No valid correspondence present.");
    }

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
    int i = 0;
    for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
            sxy_ptr[j * 3 + k] = sxy_1x9[i] / mean_1x7[6];
            ++i;
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

template <typename scalar_t>
void ComputeInformationMatrixKernelCPU(const scalar_t *target_points_ptr,
                                       const int64_t *correspondence_indices,
                                       const int n,
                                       scalar_t *global_sum) {
    // As, AtA is a symmetric matrix, we only need 21 elements instead of 36.
    std::vector<scalar_t> AtA(21, 0.0);

#ifdef _WIN32
    std::vector<scalar_t> zeros_21(21, 0.0);
    AtA = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, n), zeros_21,
            [&](tbb::blocked_range<int> r, std::vector<scalar_t> A_reduction) {
                for (int workload_idx = r.begin(); workload_idx < r.end();
                     ++workload_idx) {
#else
    scalar_t *A_reduction = AtA.data();
#pragma omp parallel for reduction(+ : A_reduction[:21]) schedule(static) num_threads(utility::EstimateMaxThreads())
    for (int workload_idx = 0; workload_idx < n; workload_idx++) {
#endif
                    scalar_t J_x[6] = {0}, J_y[6] = {0}, J_z[6] = {0};

                    bool valid = GetInformationJacobians<scalar_t>(
                            workload_idx, target_points_ptr,
                            correspondence_indices, J_x, J_y, J_z);

                    if (valid) {
                        int i = 0;
                        for (int j = 0; j < 6; ++j) {
                            for (int k = 0; k <= j; ++k) {
                                A_reduction[i] += J_x[j] * J_x[k] +
                                                  J_y[j] * J_y[k] +
                                                  J_z[j] * J_z[k];
                                ++i;
                            }
                        }
                    }
                }
#ifdef _WIN32
                return A_reduction;
            },
            // TBB: Defining reduction operation.
            [&](std::vector<scalar_t> a, std::vector<scalar_t> b) {
                std::vector<scalar_t> result(21);
                for (int j = 0; j < 21; ++j) {
                    result[j] = a[j] + b[j];
                }
                return result;
            });
#endif

    for (int i = 0; i < 21; ++i) {
        global_sum[i] = AtA[i];
    }
}

void ComputeInformationMatrixCPU(const core::Tensor &target_points,
                                 const core::Tensor &correspondence_indices,
                                 core::Tensor &information_matrix,
                                 const core::Dtype &dtype,
                                 const core::Device &device) {
    int n = correspondence_indices.GetLength();

    core::Tensor global_sum = core::Tensor::Zeros({21}, dtype, device);

    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
        scalar_t *global_sum_ptr = global_sum.GetDataPtr<scalar_t>();

        kernel::ComputeInformationMatrixKernelCPU(
                target_points.GetDataPtr<scalar_t>(),
                correspondence_indices.GetDataPtr<int64_t>(), n,
                global_sum_ptr);

        core::Tensor global_sum_cpu =
                global_sum.To(core::Device("CPU:0"), core::Float64);
        double *sum_ptr = global_sum_cpu.GetDataPtr<double>();

        // Information matrix is on CPU of type Float64.
        double *GTG_ptr = information_matrix.GetDataPtr<double>();

        int i = 0;
        for (int j = 0; j < 6; j++) {
            for (int k = 0; k <= j; k++) {
                GTG_ptr[j * 6 + k] = GTG_ptr[k * 6 + j] = sum_ptr[i];
                ++i;
            }
        }
    });
}

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
