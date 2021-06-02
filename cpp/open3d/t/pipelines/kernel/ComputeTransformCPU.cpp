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

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <cmath>
#include <functional>
#include <vector>

#include "open3d/core/Dispatch.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/CPULauncher.h"
#include "open3d/t/pipelines/kernel/Common.h"
#include "open3d/t/pipelines/kernel/ComputeTransformImpl.h"
#include "open3d/t/pipelines/kernel/TransformationConverter.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {

void ComputePosePointToPlaneCPU(const core::Tensor &source_points,
                                const core::Tensor &target_points,
                                const core::Tensor &target_normals,
                                const core::Tensor &corres,
                                core::Tensor &pose,
                                float &residual,
                                int &count,
                                const core::Dtype &dtype,
                                const core::Device &device) {
    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
        const scalar_t *source_points_ptr =
                source_points.GetDataPtr<scalar_t>();
        const scalar_t *target_points_ptr =
                target_points.GetDataPtr<scalar_t>();
        const scalar_t *target_normals_ptr =
                target_normals.GetDataPtr<scalar_t>();
        const int64_t *correspondence_indices = corres.GetDataPtr<int64_t>();

        int n = corres.GetLength();

        core::Tensor A_reduction =
                kernel::Get6x6CompressedLinearTensor<scalar_t>(
                        source_points_ptr, target_points_ptr,
                        target_normals_ptr, correspondence_indices, n);

        DecodeAndSolve6x6(A_reduction, pose, residual, count);
    });
}

template <typename scalar_t>
static void Get3x3SxyLinearSystem(const scalar_t *source_points_ptr,
                                  const scalar_t *target_points_ptr,
                                  const int64_t *correspondence_indices,
                                  const int &n,
                                  const core::Dtype &dtype,
                                  const core::Device &device,
                                  core::Tensor &Sxy,
                                  core::Tensor &mean_t,
                                  core::Tensor &mean_s,
                                  int &count) {
    // Calculating mean_s and mean_t, which are mean(x, y, z) of source and
    // target points respectively.
    std::vector<double> mean_1x7(7, 0.0);
    // Identity element for running_total reduction variable: zeros_6.
    std::vector<double> zeros_7(7, 0.0);

    mean_1x7 = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, n), zeros_7,
            [&](tbb::blocked_range<int> r, std::vector<double> mean_reduction) {
                for (int workload_idx = r.begin(); workload_idx < r.end();
                     workload_idx++) {
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
            [&](std::vector<double> a, std::vector<double> b) {
                std::vector<double> result(7);
                for (int j = 0; j < 7; j++) {
                    result[j] = a[j] + b[j];
                }
                return result;
            });

    for (int i = 0; i < 6; i++) {
        mean_1x7[i] = mean_1x7[i] / mean_1x7[6];
    }

    // Calculating the Sxy for SVD.
    std::vector<double> sxy_1x9(9, 0.0);
    // Identity element for running total reduction variable: zeros_9.
    std::vector<double> zeros_9(9, 0.0);

    sxy_1x9 = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, n), zeros_9,
            [&](tbb::blocked_range<int> r,
                std::vector<double> sxy_1x9_reduction) {
                for (int workload_idx = r.begin(); workload_idx < r.end();
                     workload_idx++) {
                    if (correspondence_indices[workload_idx] != -1) {
                        for (int i = 0; i < 9; i++) {
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
            [&](std::vector<double> a, std::vector<double> b) {
                std::vector<double> result(9);
                for (int j = 0; j < 9; j++) {
                    result[j] = a[j] + b[j];
                }
                return result;
            });

    mean_s = core::Tensor::Empty({1, 3}, dtype, device);
    scalar_t *mean_s_ptr = mean_s.GetDataPtr<scalar_t>();

    mean_t = core::Tensor::Empty({1, 3}, dtype, device);
    scalar_t *mean_t_ptr = mean_t.GetDataPtr<scalar_t>();

    Sxy = core::Tensor::Empty({3, 3}, dtype, device);
    scalar_t *sxy_ptr = Sxy.GetDataPtr<scalar_t>();

    // Getting Tensor Sxy {3,3}, mean_s {3,1} and mean_t {3} from temporary
    // reduction variables. The shapes of mean_s and mean_t are such, because it
    // will be required in equation:
    // t = mean_s - R.Matmul(mean_t.T()).Reshape({-1}).
    for (int i = 0, j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
            sxy_ptr[j * 3 + k] = sxy_1x9[i++] / mean_1x7[6];
        }
        mean_s_ptr[j] = mean_1x7[j];
        mean_t_ptr[j] = mean_1x7[j + 3];
    }

    count = static_cast<int64_t>(mean_1x7[6]);
}

void ComputeRtPointToPointCPU(const core::Tensor &source_points,
                              const core::Tensor &target_points,
                              const core::Tensor &corres,
                              core::Tensor &R,
                              core::Tensor &t,
                              int &count,
                              const core::Dtype &dtype,
                              const core::Device &device) {
    core::Tensor Sxy, mean_t, mean_s;

    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
        const scalar_t *source_points_ptr =
                source_points.GetDataPtr<scalar_t>();
        const scalar_t *target_points_ptr =
                target_points.GetDataPtr<scalar_t>();
        const int64_t *correspondence_indices = corres.GetDataPtr<int64_t>();

        int n = source_points.GetLength();

        Get3x3SxyLinearSystem(source_points_ptr, target_points_ptr,
                              correspondence_indices, n, dtype, device, Sxy,
                              mean_t, mean_s, count);
    });

    core::Tensor U, D, VT;
    std::tie(U, D, VT) = Sxy.SVD();
    core::Tensor S = core::Tensor::Eye(3, dtype, device);
    if (U.Det() * (VT.T()).Det() < 0) {
        S[-1][-1] = -1;
    }

    R = U.Matmul(S.Matmul(VT));
    t = (mean_t.Reshape({-1}) - R.Matmul(mean_s.T()).Reshape({-1})).To(dtype);
}

// void ComputePosePointToPlaneCPU(const float *source_points_ptr,
//                                 const float *target_points_ptr,
//                                 const float *target_normals_ptr,
//                                 const int64_t *correspondences_first,
//                                 const int64_t *correspondences_second,
//                                 const int n,
//                                 core::Tensor &pose,
//                                 const core::Dtype &dtype,
//                                 const core::Device &device) {
//     // As, ATA is a symmetric matrix, we only need 21 elements instead of 36.
//     // ATB is of shape {6,1}. Combining both, A_1x29 is a temp. storage
//     // with [0:20] elements as ATA and [21:26] elements as ATB.
//     // [27] is for residual (TODO @rishabh, use residual for robust kernel
//     // support), [28] is for inlier count.
//     std::vector<float> A_1x29(29, 0.0);

// #ifdef _WIN32
//     std::vector<float> zeros_29(29, 0.0);
//     A_1x29 = tbb::parallel_reduce(
//             tbb::blocked_range<int>(0, n), zeros_29,
//             [&](tbb::blocked_range<int> r, std::vector<float> A_reduction) {
//                 for (int workload_idx = r.begin(); workload_idx < r.end();
//                      workload_idx++) {
// #else
//     float *A_reduction = A_1x29.data();
// #pragma omp parallel for reduction(+ : A_reduction[:29]) schedule(static)
//     for (int workload_idx = 0; workload_idx < n; workload_idx++) {
// #endif
//                     float J[6] = {0};
//                     float r = 0;

//                     bool valid = GetJacobianPointToPlane(
//                             workload_idx, source_points_ptr,
//                             target_points_ptr, target_normals_ptr,
//                             correspondences_first, correspondences_second, J,
//                             r);

//                     if (valid) {
//                         for (int i = 0, j = 0; j < 6; j++) {
//                             for (int k = 0; k <= j; k++) {
//                                 // ATA_ {1,21}, as ATA {6,6} is a symmetric
//                                 // matrix.
//                                 A_reduction[i] += J[j] * J[k];
//                                 i++;
//                             }
//                             // ATB {6,1}.
//                             A_reduction[21 + j] += J[j] * r;
//                         }
//                         A_reduction[27] = r * r;
//                         A_reduction[28] += 1;
//                     }
//                 }
// #ifdef _WIN32
//                 return A_reduction;
//             },
//             // TBB: Defining reduction operation.
//             [&](std::vector<float> a, std::vector<float> b) {
//                 std::vector<float> result(29);
//                 for (int j = 0; j < 29; j++) {
//                     result[j] = a[j] + b[j];
//                 }
//                 return result;
//             });
// #endif

//     core::Tensor A_reduction_tensor(A_1x29, {1, 29}, core::Dtype::Float32,
//                                     device);

//     // TODO (@rishabh), residual will be used for adding robust kernel
//     support. float residual; int inlier_count;
//     // Compute linear system on CPU as Float64.
//     DecodeAndSolve6x6(A_reduction_tensor, pose, residual, inlier_count);
// }

// void ComputeRtPointToPointCPU(const float *source_points_ptr,
//                               const float *target_points_ptr,
//                               const int64_t *correspondences_first,
//                               const int64_t *correspondences_second,
//                               const int n,
//                               core::Tensor &R,
//                               core::Tensor &t,
//                               const core::Dtype dtype,
//                               const core::Device device) {
//     // Calculating mean_s and mean_t, which are mean(x, y, z) of source and
//     // target points respectively.
//     std::vector<float> mean_1x6(6, 0.0);

// #ifdef _WIN32
//     std::vector<float> zeros_6(6, 0.0);
//     mean_1x6 = tbb::parallel_reduce(
//             tbb::blocked_range<int>(0, n), zeros_6,
//             [&](tbb::blocked_range<int> r, std::vector<float> mean_reduction)
//             {
//                 for (int workload_idx = r.begin(); workload_idx < r.end();
//                      workload_idx++) {
// #else
//     float *mean_reduction = mean_1x6.data();
// #pragma omp parallel for reduction(+ : mean_reduction[:6]) schedule(static)
//     for (int workload_idx = 0; workload_idx < n; workload_idx++) {
// #endif
//                     for (int i = 0; i < 3; i++) {
//                         mean_reduction[i] += source_points_ptr
//                                 [3 * correspondences_first[workload_idx] +
//                                 i];
//                         mean_reduction[i + 3] += target_points_ptr
//                                 [3 * correspondences_second[workload_idx] +
//                                 i];
//                     }
//                 }
// #ifdef _WIN32
//                 return mean_reduction;
//             },
//             // TBB: Defining reduction operation.
//             [&](std::vector<float> a, std::vector<float> b) {
//                 std::vector<float> result(6);
//                 for (int j = 0; j < 6; j++) {
//                     result[j] = a[j] + b[j];
//                 }
//                 return result;
//             });
// #endif

//     float num_correspondences = static_cast<float>(n);
//     for (int i = 0; i < 6; i++) {
//         mean_1x6[i] = mean_1x6[i] / num_correspondences;
//     }

//     // Calculating the Sxy for SVD.
//     std::vector<float> sxy_1x9(9, 0.0);

// #ifdef _WIN32
//     std::vector<float> zeros_9(9, 0.0);
//     sxy_1x9 = tbb::parallel_reduce(
//             tbb::blocked_range<int>(0, n), zeros_9,
//             [&](tbb::blocked_range<int> r,
//                 std::vector<float> sxy_1x9_reduction) {
//                 for (int workload_idx = r.begin(); workload_idx < r.end();
//                      workload_idx++) {
// #else
//     float *sxy_1x9_reduction = sxy_1x9.data();
// #pragma omp parallel for reduction(+ : sxy_1x9_reduction[:9])
// schedule(static) collapse(2)
//     for (int workload_idx = 0; workload_idx < n; workload_idx++) {
// #endif
//                     for (int i = 0; i < 9; i++) {
//                         const int row = i % 3;
//                         const int col = i / 3;
//                         const int source_idx =
//                                 3 * correspondences_first[workload_idx] +
//                                 row;
//                         const int target_idx =
//                                 3 * correspondences_second[workload_idx] +
//                                 col;
//                         sxy_1x9_reduction[i] +=
//                         (source_points_ptr[source_idx] -
//                                                  mean_1x6[row]) *
//                                                 (target_points_ptr[target_idx]
//                                                 -
//                                                  mean_1x6[3 + col]);
//                     }
//                 }
// #ifdef _WIN32
//                 return sxy_1x9_reduction;
//             },
//             // TBB: Defining reduction operation.
//             [&](std::vector<float> a, std::vector<float> b) {
//                 std::vector<float> result(9);
//                 for (int j = 0; j < 9; j++) {
//                     result[j] = a[j] + b[j];
//                 }
//                 return result;
//             });
// #endif

//     // Compute linear system on CPU as Float64.
//     core::Device host("CPU:0");
//     core::Tensor mean_s =
//             core::Tensor::Empty({1, 3}, core::Dtype::Float64, host);
//     double *mean_s_ptr = mean_s.GetDataPtr<double>();

//     core::Tensor mean_t =
//             core::Tensor::Empty({1, 3}, core::Dtype::Float64, host);
//     double *mean_t_ptr = mean_t.GetDataPtr<double>();

//     core::Tensor Sxy = core::Tensor::Empty({3, 3}, core::Dtype::Float64,
//     host); double *sxy_ptr = Sxy.GetDataPtr<double>();

//     // Getting Tensor Sxy {3,3}, mean_s {3,1} and mean_t {3} from temporary
//     // reduction variables. The shapes of mean_s and mean_t are such, because
//     it
//     // will be required in equation:
//     // t = mean_s - R.Matmul(mean_t.T()).Reshape({-1}).
//     for (int i = 0, j = 0; j < 3; j++) {
//         for (int k = 0; k < 3; k++) {
//             sxy_ptr[j * 3 + k] = sxy_1x9[i++] / num_correspondences;
//         }
//         mean_s_ptr[j] = mean_1x6[j];
//         mean_t_ptr[j] = mean_1x6[j + 3];
//     }

//     core::Tensor U, D, VT;
//     std::tie(U, D, VT) = Sxy.SVD();
//     core::Tensor S = core::Tensor::Eye(3, core::Dtype::Float64, host);
//     if (U.Det() * (VT.T()).Det() < 0) {
//         S[-1][-1] = -1;
//     }

//     R = U.Matmul(S.Matmul(VT));
//     t = mean_t.Reshape({-1}) - R.Matmul(mean_s.T()).Reshape({-1});
// }

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
