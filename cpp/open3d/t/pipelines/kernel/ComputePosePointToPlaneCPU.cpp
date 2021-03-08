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
#include "open3d/t/pipelines/kernel/ComputePosePointToPlaneImp.h"
#include "open3d/t/pipelines/kernel/TransformationConverter.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {

void ComputePosePointToPlaneCPU(const float *source_points_ptr,
                                const float *target_points_ptr,
                                const float *target_normals_ptr,
                                const int64_t *correspondence_first,
                                const int64_t *correspondence_second,
                                const int n,
                                core::Tensor &pose,
                                const core::Dtype &dtype,
                                const core::Device &device) {
    // As, ATA is a symmetric matrix, we only need 21 elements instead of 36.
    // ATB is of shape {6,1}. Combining both, A_1x27 is a temp. storage
    // with [0:21] elements as ATA and [21:27] elements as ATB.
    std::vector<double> A_1x27(27, 0.0);
#ifdef _WIN32
    // Identity element for running_total reduction variable: zeros_27.
    std::vector<double> zeros_27(27, 0.0);
    // For TBB reduction, A_ is a reduction variable of type vector<double>.
    A_1x27 = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, n), zeros_27,
            [&](tbb::blocked_range<int> r, std::vector<double> A_) {
                for (int workload_idx = r.begin(); workload_idx < r.end();
                     workload_idx++) {
#else
    // For OpenMP reduction, A_ is a double pointer to A_1x27.
    double *A_ = A_1x27.data();
#pragma omp parallel for reduction(+ : A_[:27])
    for (int64_t workload_idx = 0; workload_idx < n; ++workload_idx) {
#endif
                    const int64_t &source_index =
                            3 * correspondence_first[workload_idx];
                    const int64_t &target_index =
                            3 * correspondence_second[workload_idx];

                    const float &sx = (source_points_ptr[source_index + 0]);
                    const float &sy = (source_points_ptr[source_index + 1]);
                    const float &sz = (source_points_ptr[source_index + 2]);
                    const float &tx = (target_points_ptr[target_index + 0]);
                    const float &ty = (target_points_ptr[target_index + 1]);
                    const float &tz = (target_points_ptr[target_index + 2]);
                    const float &nx = (target_normals_ptr[target_index + 0]);
                    const float &ny = (target_normals_ptr[target_index + 1]);
                    const float &nz = (target_normals_ptr[target_index + 2]);

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
                            A_[i] += ai[j] * ai[k];
                            i++;
                        }
                        // ATB {6,1}.
                        A_[21 + j] += ai[j] * bi_neg;
                    }
                }
#ifdef _WIN32
                return A_;
            },
            // TBB: Defining reduction operation.
            [&](std::vector<double> a, std::vector<double> b) {
                std::vector<double> result(27);
                for (int j = 0; j < 27; j++) {
                    result[j] = a[j] + b[j];
                }
                return result;
            });
#endif

    core::Tensor ATA =
            core::Tensor::Empty({6, 6}, core::Dtype::Float64, device);
    double *ata_ptr = ATA.GetDataPtr<double>();

    // ATB_neg is -(ATB), as bi_neg is used in kernel instead of bi,
    // where  bi = [source_points - target_points].(target_normals).
    core::Tensor ATB_neg =
            core::Tensor::Empty({6, 1}, core::Dtype::Float64, device);
    double *atb_ptr = ATB_neg.GetDataPtr<double>();

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

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
