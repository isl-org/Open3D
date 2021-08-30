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

#include <cuda.h>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/pipelines/kernel/Reduction6x6Impl.cuh"
#include "open3d/t/pipelines/kernel/RegistrationImpl.h"
#include "open3d/t/pipelines/kernel/TransformationConverter.h"
#include "open3d/t/pipelines/registration/RobustKernel.h"
#include "open3d/t/pipelines/registration/RobustKernelImpl.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {

const int kThread1DUnit = 256;

template <typename scalar_t, typename func_t>
__global__ void ComputePosePointToPlaneKernelCUDA(
        const scalar_t *source_points_ptr,
        const scalar_t *target_points_ptr,
        const scalar_t *target_normals_ptr,
        const int64_t *correspondence_indices,
        const int n,
        scalar_t *global_sum,
        func_t GetWeightFromRobustKernel) {
    __shared__ scalar_t local_sum0[kThread1DUnit];
    __shared__ scalar_t local_sum1[kThread1DUnit];
    __shared__ scalar_t local_sum2[kThread1DUnit];

    const int tid = threadIdx.x;

    local_sum0[tid] = 0;
    local_sum1[tid] = 0;
    local_sum2[tid] = 0;

    const int workload_idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (workload_idx >= n) return;

    scalar_t J_ij[6] = {0}, reduction[29] = {0};
    scalar_t r = 0;

    bool valid = GetJacobianPointToPlane<scalar_t>(
            workload_idx, source_points_ptr, target_points_ptr,
            target_normals_ptr, correspondence_indices, J_ij, r);

    scalar_t w = GetWeightFromRobustKernel(r);

    if (valid) {
        // Dump J, r into JtJ and Jtr
        int i = 0;
        for (int j = 0; j < 6; ++j) {
            for (int k = 0; k <= j; ++k) {
                reduction[i] += J_ij[j] * w * J_ij[k];
                ++i;
            }
            reduction[21 + j] += J_ij[j] * w * r;
        }
        reduction[27] += r;
        reduction[28] += 1;
    }

    ReduceSum6x6LinearSystem<scalar_t, kThread1DUnit>(tid, valid, reduction,
                                                      local_sum0, local_sum1,
                                                      local_sum2, global_sum);
}

void ComputePosePointToPlaneCUDA(const core::Tensor &source_points,
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
    const dim3 blocks((n + kThread1DUnit - 1) / kThread1DUnit);
    const dim3 threads(kThread1DUnit);

    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
        scalar_t *global_sum_ptr = global_sum.GetDataPtr<scalar_t>();

        DISPATCH_ROBUST_KERNEL_FUNCTION(
                kernel.type_, scalar_t, kernel.scaling_parameter_,
                kernel.shape_parameter_, [&]() {
                    ComputePosePointToPlaneKernelCUDA<<<
                            blocks, threads, 0, core::cuda::GetStream()>>>(
                            source_points.GetDataPtr<scalar_t>(),
                            target_points.GetDataPtr<scalar_t>(),
                            target_normals.GetDataPtr<scalar_t>(),
                            correspondence_indices.GetDataPtr<int64_t>(), n,
                            global_sum_ptr, GetWeightFromRobustKernel);
                });
    });

    core::cuda::Synchronize();

    DecodeAndSolve6x6(global_sum, pose, residual, inlier_count);
}

template <typename scalar_t, typename funct_t>
__global__ void ComputePoseColoredICPKernelCUDA(
        const scalar_t *source_points_ptr,
        const scalar_t *source_colors_ptr,
        const scalar_t *target_points_ptr,
        const scalar_t *target_normals_ptr,
        const scalar_t *target_colors_ptr,
        const scalar_t *target_color_gradients_ptr,
        const int64_t *correspondence_indices,
        const scalar_t sqrt_lambda_geometric,
        const scalar_t sqrt_lambda_photometric,
        const int n,
        scalar_t *global_sum,
        funct_t GetWeightFromRobustKernel) {
    __shared__ scalar_t local_sum0[kThread1DUnit];
    __shared__ scalar_t local_sum1[kThread1DUnit];
    __shared__ scalar_t local_sum2[kThread1DUnit];

    const int tid = threadIdx.x;

    local_sum0[tid] = 0;
    local_sum1[tid] = 0;
    local_sum2[tid] = 0;

    const int workload_idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (workload_idx >= n) return;

    scalar_t J_G[6] = {0}, J_I[6] = {0}, reduction[29] = {0};
    scalar_t r_G = 0, r_I = 0;

    bool valid = GetJacobianColoredICP<scalar_t>(
            workload_idx, source_points_ptr, source_colors_ptr,
            target_points_ptr, target_normals_ptr, target_colors_ptr,
            target_color_gradients_ptr, correspondence_indices,
            sqrt_lambda_geometric, sqrt_lambda_photometric, J_G, J_I, r_G, r_I);

    scalar_t w_G = GetWeightFromRobustKernel(r_G);
    scalar_t w_I = GetWeightFromRobustKernel(r_I);

    if (valid) {
        // Dump J, r into JtJ and Jtr
        int i = 0;
        for (int j = 0; j < 6; ++j) {
            for (int k = 0; k <= j; ++k) {
                reduction[i] += J_G[j] * w_G * J_G[k] + J_I[j] * w_I * J_I[k];
                ++i;
            }
            reduction[21 + j] += J_G[j] * w_G * r_G + J_I[j] * w_I * r_I;
        }
        reduction[27] += r_G * r_G + r_I * r_I;
        reduction[28] += 1;
    }

    ReduceSum6x6LinearSystem<scalar_t, kThread1DUnit>(tid, valid, reduction,
                                                      local_sum0, local_sum1,
                                                      local_sum2, global_sum);
}

void ComputePoseColoredICPCUDA(const core::Tensor &source_points,
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
    const dim3 blocks((n + kThread1DUnit - 1) / kThread1DUnit);
    const dim3 threads(kThread1DUnit);

    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
        scalar_t sqrt_lambda_geometric =
                static_cast<scalar_t>(sqrt(lambda_geometric));
        scalar_t sqrt_lambda_photometric =
                static_cast<scalar_t>(sqrt(1.0 - lambda_geometric));

        DISPATCH_ROBUST_KERNEL_FUNCTION(
                kernel.type_, scalar_t, kernel.scaling_parameter_,
                kernel.shape_parameter_, [&]() {
                    ComputePoseColoredICPKernelCUDA<<<
                            blocks, threads, 0, core::cuda::GetStream()>>>(
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

    core::cuda::Synchronize();

    DecodeAndSolve6x6(global_sum, pose, residual, inlier_count);
}

template <typename scalar_t>
__global__ void ComputeInformationMatrixKernelCUDA(
        const scalar_t *target_points_ptr,
        const int64_t *correspondence_indices,
        const int n,
        scalar_t *global_sum) {
    __shared__ scalar_t local_sum0[kThread1DUnit];
    __shared__ scalar_t local_sum1[kThread1DUnit];
    __shared__ scalar_t local_sum2[kThread1DUnit];

    const int tid = threadIdx.x;

    local_sum0[tid] = 0;
    local_sum1[tid] = 0;
    local_sum2[tid] = 0;

    const int workload_idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (workload_idx >= n) return;

    scalar_t J_x[6] = {0}, J_y[6] = {0}, J_z[6] = {0}, reduction[21] = {0};

    bool valid = GetInformationJacobians<scalar_t>(
            workload_idx, target_points_ptr, correspondence_indices, J_x, J_y,
            J_z);

    if (valid) {
        int i = 0;
        for (int j = 0; j < 6; ++j) {
            for (int k = 0; k <= j; ++k) {
                reduction[i] +=
                        J_x[j] * J_x[k] + J_y[j] * J_y[k] + J_z[j] * J_z[k];
                ++i;
            }
        }
    }

    ReduceSum6x6InformationJacobian<scalar_t, kThread1DUnit>(
            tid, valid, reduction, local_sum0, local_sum1, local_sum2,
            global_sum);
}

void ComputeInformationMatrixCUDA(const core::Tensor &target_points,
                                  const core::Tensor &correspondence_indices,
                                  core::Tensor &information_matrix,
                                  const core::Dtype &dtype,
                                  const core::Device &device) {
    int n = correspondence_indices.GetLength();

    core::Tensor global_sum = core::Tensor::Zeros({21}, dtype, device);
    const dim3 blocks((n + kThread1DUnit - 1) / kThread1DUnit);
    const dim3 threads(kThread1DUnit);

    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
        scalar_t *global_sum_ptr = global_sum.GetDataPtr<scalar_t>();

        ComputeInformationMatrixKernelCUDA<<<blocks, threads, 0,
                                             core::cuda::GetStream()>>>(
                target_points.GetDataPtr<scalar_t>(),
                correspondence_indices.GetDataPtr<int64_t>(), n,
                global_sum_ptr);

        core::cuda::Synchronize();

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
