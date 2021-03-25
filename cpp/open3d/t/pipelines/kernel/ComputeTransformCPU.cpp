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

#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/CPULauncher.h"
#include "open3d/t/pipelines/kernel/ComputeTransformImpl.h"
#include "open3d/t/pipelines/kernel/TransformationConverter.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {

void ComputePosePointToPlaneCPU(const float *source_points_ptr,
                                const float *target_points_ptr,
                                const float *target_normals_ptr,
                                const int64_t *correspondences_first,
                                const int64_t *correspondences_second,
                                const int n,
                                core::Tensor &pose,
                                const core::Dtype &dtype,
                                const core::Device &device) {
    // As, ATA is a symmetric matrix, we only need 21 elements instead of 36.
    // ATB is of shape {6,1}. Combining both, A_1x27 is a temp. storage
    // with [0:21] elements as ATA and [21:27] elements as ATB.
    std::vector<double> A_1x27(27, 0.0);
    // Identity element for running_total reduction variable: zeros_27.
    std::vector<double> zeros_27(27, 0.0);

    A_1x27 = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, n), zeros_27,
            [&](tbb::blocked_range<int> r, std::vector<double> A_reduction) {
                for (int workload_idx = r.begin(); workload_idx < r.end();
                     workload_idx++) {
                    const int64_t &source_idx =
                            3 * correspondences_first[workload_idx];
                    const int64_t &target_idx =
                            3 * correspondences_second[workload_idx];

                    const float &sx = source_points_ptr[source_idx + 0];
                    const float &sy = source_points_ptr[source_idx + 1];
                    const float &sz = source_points_ptr[source_idx + 2];
                    const float &tx = target_points_ptr[target_idx + 0];
                    const float &ty = target_points_ptr[target_idx + 1];
                    const float &tz = target_points_ptr[target_idx + 2];
                    const float &nx = target_normals_ptr[target_idx + 0];
                    const float &ny = target_normals_ptr[target_idx + 1];
                    const float &nz = target_normals_ptr[target_idx + 2];

                    const double bi_neg =
                            (tx - sx) * nx + (ty - sy) * ny + (tz - sz) * nz;
                    const double ai[] = {(nz * sy - ny * sz),
                                         (nx * sz - nz * sx),
                                         (ny * sx - nx * sy),
                                         nx,
                                         ny,
                                         nz};

                    for (int i = 0, j = 0; j < 6; j++) {
                        for (int k = 0; k <= j; k++) {
                            // ATA_ {1,21}, as ATA {6,6} is a symmetric matrix.
                            A_reduction[i] += ai[j] * ai[k];
                            i++;
                        }
                        // ATB {6,1}.
                        A_reduction[21 + j] += ai[j] * bi_neg;
                    }
                }
                return A_reduction;
            },
            // TBB: Defining reduction operation.
            [&](std::vector<double> a, std::vector<double> b) {
                std::vector<double> result(27);
                for (int j = 0; j < 27; j++) {
                    result[j] = a[j] + b[j];
                }
                return result;
            });

    core::Tensor ATA =
            core::Tensor::Empty({6, 6}, core::Dtype::Float32, device);
    float *ata_ptr = ATA.GetDataPtr<float>();

    // ATB_neg is -(ATB), as bi_neg is used in kernel instead of bi,
    // where  bi = [source_points - target_points].(target_normals).
    core::Tensor ATB_neg =
            core::Tensor::Empty({6, 1}, core::Dtype::Float32, device);
    float *atb_ptr = ATB_neg.GetDataPtr<float>();

    // ATA_ {1,21} to ATA {6,6}.
    for (int i = 0, j = 0; j < 6; j++) {
        for (int k = 0; k <= j; k++) {
            ata_ptr[j * 6 + k] = A_1x27[i];
            ata_ptr[k * 6 + j] = A_1x27[i];
            i++;
        }
        atb_ptr[j] = A_1x27[21 + j];
    }

    // ATA(6,6) . Pose(6,1) = -ATB(6,1).
    pose = ATA.Solve(ATB_neg).Reshape({-1}).To(dtype);
}

void ComputeRtPointToPointCPU(const float *source_points_ptr,
                              const float *target_points_ptr,
                              const int64_t *correspondences_first,
                              const int64_t *correspondences_second,
                              const int n,
                              core::Tensor &R,
                              core::Tensor &t,
                              const core::Dtype dtype,
                              const core::Device device) {
    // Calculating mean_s and mean_t, which are mean(x, y, z) of source and
    // target points respectively.
    std::vector<double> mean_1x6(6, 0.0);
    // Identity element for running_total reduction variable: zeros_6.
    std::vector<double> zeros_6(6, 0.0);

    mean_1x6 = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, n), zeros_6,
            [&](tbb::blocked_range<int> r, std::vector<double> mean_reduction) {
                for (int workload_idx = r.begin(); workload_idx < r.end();
                     workload_idx++) {
                    for (int i = 0; i < 3; i++) {
                        mean_reduction[i] += source_points_ptr
                                [3 * correspondences_first[workload_idx] + i];
                        mean_reduction[i + 3] += target_points_ptr
                                [3 * correspondences_second[workload_idx] + i];
                    }
                }
                return mean_reduction;
            },
            // TBB: Defining reduction operation.
            [&](std::vector<double> a, std::vector<double> b) {
                std::vector<double> result(6);
                for (int j = 0; j < 6; j++) {
                    result[j] = a[j] + b[j];
                }
                return result;
            });

    double num_correspondences = static_cast<double>(n);
    for (int i = 0; i < 6; i++) {
        mean_1x6[i] = mean_1x6[i] / num_correspondences;
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
                    for (int i = 0; i < 9; i++) {
                        const int row = i % 3;
                        const int col = i / 3;
                        const int source_idx =
                                3 * correspondences_first[workload_idx] + row;
                        const int target_idx =
                                3 * correspondences_second[workload_idx] + col;
                        sxy_1x9_reduction[i] += (source_points_ptr[source_idx] -
                                                 mean_1x6[row]) *
                                                (target_points_ptr[target_idx] -
                                                 mean_1x6[3 + col]);
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

    core::Tensor mean_s =
            core::Tensor::Empty({1, 3}, core::Dtype::Float32, device);
    float *mean_s_ptr = mean_s.GetDataPtr<float>();

    core::Tensor mean_t =
            core::Tensor::Empty({1, 3}, core::Dtype::Float32, device);
    float *mean_t_ptr = mean_t.GetDataPtr<float>();

    core::Tensor Sxy =
            core::Tensor::Empty({3, 3}, core::Dtype::Float32, device);
    float *sxy_ptr = Sxy.GetDataPtr<float>();

    // Getting Tensor Sxy {3,3}, mean_s {3,1} and mean_t {3} from temporary
    // reduction variables. The shapes of mean_s and mean_t are such, because it
    // will be required in equation:
    // t = mean_s - R.Matmul(mean_t.T()).Reshape({-1}).
    for (int i = 0, j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
            sxy_ptr[j * 3 + k] = sxy_1x9[i++] / num_correspondences;
        }
        mean_s_ptr[j] = mean_1x6[j];
        mean_t_ptr[j] = mean_1x6[j + 3];
    }

    core::Tensor U, D, VT;
    std::tie(U, D, VT) = Sxy.SVD();
    core::Tensor S = core::Tensor::Eye(3, dtype, device);
    if (U.Det() * (VT.T()).Det() < 0) {
        S[-1][-1] = -1;
    }

    R = U.Matmul(S.Matmul(VT));
    t = mean_t.Reshape({-1}) - R.Matmul(mean_s.T()).Reshape({-1});
}

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
