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

#include "open3d/t/pipelines/kernel/Common.h"

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include "open3d/t/pipelines/kernel/RegistrationImpl.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {

template <typename scalar_t>
core::Tensor Get6x6CompressedLinearTensor(
        const scalar_t *source_points_ptr,
        const scalar_t *target_points_ptr,
        const scalar_t *target_normals_ptr,
        const int64_t *correspondences_first,
        const scalar_t *correspondences_second,
        const int n) {
    utility::LogError(" Get6x6CompressedLinearTensor: Datatype not supported.");
}

template <>
core::Tensor Get6x6CompressedLinearTensor<float>(
        const float *source_points_ptr,
        const float *target_points_ptr,
        const float *target_normals_ptr,
        const int64_t *correspondences_first,
        const float *correspondences_second,
        const int n) {
    // As, ATA is a symmetric matrix, we only need 21 elements instead of 36.
    // ATB is of shape {6,1}. Combining both, A_1x27 is a temp. storage
    // with [0:21] elements as ATA and [21:27] elements as ATB.
    std::vector<double> A_1x29(29, 0.0);
    // Identity element for running_total reduction variable: zeros_27.
    std::vector<double> zeros_29(29, 0.0);

    A_1x29 = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, n), zeros_29,
            [&](tbb::blocked_range<int> r, std::vector<double> A_reduction) {
                for (int workload_idx = r.begin(); workload_idx < r.end();
                     workload_idx++) {
                    float J_ij[6];
                    float r;

                    bool valid = kernel::registration::GetJacobianPointToPlane<
                            float>(workload_idx, source_points_ptr,
                                   target_points_ptr, target_normals_ptr,
                                   correspondences_first, J_ij, r);

                    if (valid) {
                        A_reduction[0] += J_ij[0] * J_ij[0];
                        A_reduction[1] += J_ij[1] * J_ij[0];
                        A_reduction[2] += J_ij[1] * J_ij[1];
                        A_reduction[3] += J_ij[2] * J_ij[0];
                        A_reduction[4] += J_ij[2] * J_ij[1];
                        A_reduction[5] += J_ij[2] * J_ij[2];
                        A_reduction[6] += J_ij[3] * J_ij[0];
                        A_reduction[7] += J_ij[3] * J_ij[1];
                        A_reduction[8] += J_ij[3] * J_ij[2];
                        A_reduction[9] += J_ij[3] * J_ij[3];
                        A_reduction[10] += J_ij[4] * J_ij[0];
                        A_reduction[11] += J_ij[4] * J_ij[1];
                        A_reduction[12] += J_ij[4] * J_ij[2];
                        A_reduction[13] += J_ij[4] * J_ij[3];
                        A_reduction[14] += J_ij[4] * J_ij[4];
                        A_reduction[15] += J_ij[5] * J_ij[0];
                        A_reduction[16] += J_ij[5] * J_ij[1];
                        A_reduction[17] += J_ij[5] * J_ij[2];
                        A_reduction[18] += J_ij[5] * J_ij[3];
                        A_reduction[19] += J_ij[5] * J_ij[4];
                        A_reduction[20] += J_ij[5] * J_ij[5];

                        A_reduction[21] += J_ij[0] * r;
                        A_reduction[22] += J_ij[1] * r;
                        A_reduction[23] += J_ij[2] * r;
                        A_reduction[24] += J_ij[3] * r;
                        A_reduction[25] += J_ij[4] * r;
                        A_reduction[26] += J_ij[5] * r;

                        A_reduction[27] += correspondences_second[workload_idx];
                        A_reduction[28] += 1;
                    }
                }
                return A_reduction;
            },
            // TBB: Defining reduction operation.
            [&](const std::vector<double> &a, const std::vector<double> &b) {
                std::vector<double> result(29);
                for (int j = 0; j < 29; j++) {
                    result[j] = a[j] + b[j];
                }
                return result;
            });
    return core::Tensor(A_1x29, {29}, core::Dtype::Float64);
}

template <>
core::Tensor Get6x6CompressedLinearTensor<double>(
        const double *source_points_ptr,
        const double *target_points_ptr,
        const double *target_normals_ptr,
        const int64_t *correspondences_first,
        const double *correspondences_second,
        const int n) {
    // As, ATA is a symmetric matrix, we only need 21 elements instead of 36.
    // ATB is of shape {6,1}. Combining both, A_1x27 is a temp. storage
    // with [0:21] elements as ATA and [21:27] elements as ATB.
    std::vector<double> A_1x29(29, 0.0);
    // Identity element for running_total reduction variable: zeros_27.
    std::vector<double> zeros_29(29, 0.0);

    A_1x29 = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, n), zeros_29,
            [&](tbb::blocked_range<int> r, std::vector<double> A_reduction) {
                for (int workload_idx = r.begin(); workload_idx < r.end();
                     workload_idx++) {
                    double J_ij[6];
                    double r;

                    bool valid = kernel::registration::GetJacobianPointToPlane<
                            double>(workload_idx, source_points_ptr,
                                    target_points_ptr, target_normals_ptr,
                                    correspondences_first, J_ij, r);

                    if (valid) {
                        A_reduction[0] += J_ij[0] * J_ij[0];
                        A_reduction[1] += J_ij[1] * J_ij[0];
                        A_reduction[2] += J_ij[1] * J_ij[1];
                        A_reduction[3] += J_ij[2] * J_ij[0];
                        A_reduction[4] += J_ij[2] * J_ij[1];
                        A_reduction[5] += J_ij[2] * J_ij[2];
                        A_reduction[6] += J_ij[3] * J_ij[0];
                        A_reduction[7] += J_ij[3] * J_ij[1];
                        A_reduction[8] += J_ij[3] * J_ij[2];
                        A_reduction[9] += J_ij[3] * J_ij[3];
                        A_reduction[10] += J_ij[4] * J_ij[0];
                        A_reduction[11] += J_ij[4] * J_ij[1];
                        A_reduction[12] += J_ij[4] * J_ij[2];
                        A_reduction[13] += J_ij[4] * J_ij[3];
                        A_reduction[14] += J_ij[4] * J_ij[4];
                        A_reduction[15] += J_ij[5] * J_ij[0];
                        A_reduction[16] += J_ij[5] * J_ij[1];
                        A_reduction[17] += J_ij[5] * J_ij[2];
                        A_reduction[18] += J_ij[5] * J_ij[3];
                        A_reduction[19] += J_ij[5] * J_ij[4];
                        A_reduction[20] += J_ij[5] * J_ij[5];

                        A_reduction[21] += J_ij[0] * r;
                        A_reduction[22] += J_ij[1] * r;
                        A_reduction[23] += J_ij[2] * r;
                        A_reduction[24] += J_ij[3] * r;
                        A_reduction[25] += J_ij[4] * r;
                        A_reduction[26] += J_ij[5] * r;

                        A_reduction[27] += correspondences_second[workload_idx];
                        A_reduction[28] += 1;
                    }
                }
                return A_reduction;
            },
            // TBB: Defining reduction operation.
            [&](const std::vector<double> &a, const std::vector<double> &b) {
                std::vector<double> result(29);
                for (int j = 0; j < 29; j++) {
                    result[j] = a[j] + b[j];
                }
                return result;
            });
    return core::Tensor(A_1x29, {29}, core::Dtype::Float64);
}

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
