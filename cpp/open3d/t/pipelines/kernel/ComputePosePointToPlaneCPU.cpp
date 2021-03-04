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
#include "open3d/utility/Timer.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {

#ifdef _WIN32
static void ComputePosePointToPlaneTBB(const float *source_points_ptr,
                                       const float *target_points_ptr,
                                       const float *target_normals_ptr,
                                       const int64_t *correspondence_first,
                                       const int64_t *correspondence_second,
                                       const int n,
                                       core::Tensor &pose,
                                       const core::Dtype &dtype,
                                       const core::Device &device) {
    // ATA of shape {N,21} that will be reduced to ATA of shape {6,6}.
    // As, ATA is a symmetric matrix, we only need 21 elements instead of 36.
    // ATB of shape {N,6} that will be reduced to ATB of shape {6,1}.
    // Combining both, A_Nx27 is a temp. storage with [0:21] elements as ATA
    // and [21:27] elements as ATB.
    core::Tensor A_Nx27 =
            core::Tensor::Empty({n, 27}, core::Dtype::Float32, device);
    float *A_Nx27_ptr = A_Nx27.GetDataPtr<float>();

    tbb::parallel_for(
            tbb::blocked_range<int>(0, n), [&](tbb::blocked_range<int> r) {
                for (int workload_idx = r.begin(); workload_idx < r.end();
                     ++workload_idx) {
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

                    const float bi =
                            (tx - sx) * nx + (ty - sy) * ny + (tz - sz) * nz;
                    const float ai[] = {(nz * sy - ny * sz),
                                        (nx * sz - nz * sx),
                                        (ny * sx - nx * sy),
                                        nx,
                                        ny,
                                        nz};

                    for (int i = 0, j = 0; j < 6; j++) {
                        for (int k = 0; k <= j; k++) {
                            // ATA {N,21}
                            A_Nx27_ptr[workload_idx * 27 + i] = ai[j] * ai[k];
                            i++;
                        }
                        // ATB {N,6}.
                        A_Nx27_ptr[workload_idx * 27 + 21 + j] = ai[j] * bi;
                    }
                }
            });

    // Reduce A {N, 27} to {1, 27}.
    std::vector<float> zeros_27(27, 0.0);
    std::vector<float> A_1x27_vec = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, n), zeros_27,
            [&](tbb::blocked_range<int> r, std::vector<float> running_total) {
                for (int i = r.begin(); i < r.end(); i++) {
                    for (int j = 0; j < 27; j++) {
                        running_total[j] += A_Nx27_ptr[i * 27 + j];
                    }
                }
                return running_total;
            },
            [&](std::vector<float> a, std::vector<float> b) {
                std::vector<float> result(27);
                for (int j = 0; j < 27; j++) {
                    result[j] = a[j] + b[j];
                }
                return result;
            });

    core::Tensor ATA =
            core::Tensor::Empty({6, 6}, core::Dtype::Float32, device);
    float *ata_ptr = ATA.GetDataPtr<float>();

    core::Tensor ATB =
            core::Tensor::Empty({6, 1}, core::Dtype::Float32, device);
    float *atb_ptr = ATB.GetDataPtr<float>();

    //   ata_1x21 is a {1,21} vector having elements of the matrix ATA such
    //     that the corresponding elemetes in ATA are like:
    //     0
    //     1   2
    //     3   4   5
    //     6   7   8   9
    //     10  11  12  13  14
    //     15  16  17  18  19  20
    //     Since, ATA is a symmertric matrix, it can be regenerated from this
    // Get the ATA matrix back.
    // ATA {1,21} to ATA {6,6}.
    for (int i = 0, j = 0; j < 6; j++) {
        for (int k = 0; k <= j; k++) {
            ata_ptr[j * 6 + k] = A_1x27_vec[i];
            ata_ptr[k * 6 + j] = A_1x27_vec[i];
            i++;
        }
        atb_ptr[j] = A_1x27_vec[j + 21];
    }

    // ATA(6,6) . Pose(6,1) = ATB(6,1)
    pose = ATA.Solve(ATB).Reshape({-1}).To(dtype);
}
#else
static void ComputePosePointToPlaneOpenMP(const float *source_points_ptr,
                                          const float *target_points_ptr,
                                          const float *target_normals_ptr,
                                          const int64_t *correspondence_first,
                                          const int64_t *correspondence_second,
                                          const int n,
                                          core::Tensor &pose,
                                          const core::Dtype &dtype,
                                          const core::Device &device) {
    core::Tensor ATA =
            core::Tensor::Empty({6, 6}, core::Dtype::Float64, device);
    core::Tensor ATA_1x21 =
            core::Tensor::Zeros({1, 21}, core::Dtype::Float64, device);
    core::Tensor ATB =
            core::Tensor::Zeros({6, 1}, core::Dtype::Float64, device);

    double *ata_ptr = ATA.GetDataPtr<double>();
    double *ata_1x21 = ATA_1x21.GetDataPtr<double>();
    double *atb_ptr = ATB.GetDataPtr<double>();

#pragma omp parallel for reduction(+ : atb_ptr[:6], ata_1x21[:21])
    for (int64_t workload_idx = 0; workload_idx < n; ++workload_idx) {
        const int64_t &source_index = 3 * correspondence_first[workload_idx];
        const int64_t &target_index = 3 * correspondence_second[workload_idx];

        const float &sx = (source_points_ptr[source_index + 0]);
        const float &sy = (source_points_ptr[source_index + 1]);
        const float &sz = (source_points_ptr[source_index + 2]);
        const float &tx = (target_points_ptr[target_index + 0]);
        const float &ty = (target_points_ptr[target_index + 1]);
        const float &tz = (target_points_ptr[target_index + 2]);
        const float &nx = (target_normals_ptr[target_index + 0]);
        const float &ny = (target_normals_ptr[target_index + 1]);
        const float &nz = (target_normals_ptr[target_index + 2]);

        const float bi = (tx - sx) * nx + (ty - sy) * ny + (tz - sz) * nz;
        const float ai[] = {(nz * sy - ny * sz),
                            (nx * sz - nz * sx),
                            (ny * sx - nx * sy),
                            nx,
                            ny,
                            nz};

        for (int i = 0, j = 0; j < 6; j++) {
            for (int k = 0; k <= j; k++) {
                // ATA_ {1,21}, as ATA {6,6} is a symmetric matrix.
                ata_1x21[i] += ai[j] * ai[k];
                i++;
            }
            // ATB {6,1}.
            atb_ptr[j] += ai[j] * bi;
        }
    }
    //   ata_1x21 is a {1,21} vector having elements of the matrix ATA such
    //     that the corresponding elemetes in ATA are like:
    //     0
    //     1   2
    //     3   4   5
    //     6   7   8   9
    //     10  11  12  13  14
    //     15  16  17  18  19  20
    //     Since, ATA is a symmertric matrix, it can be regenerated from this
    // Get the ATA matrix back.
    // ATA_ {1,21} to ATA {6,6}.
    for (int i = 0, j = 0; j < 6; j++) {
        for (int k = 0; k <= j; k++) {
            ata_ptr[j * 6 + k] = ata_1x21[i];
            ata_ptr[k * 6 + j] = ata_1x21[i];
            i++;
        }
    }

    // ATA(6,6) . Pose(6,1) = ATB(6,1)
    pose = ATA.Solve(ATB).Reshape({-1}).To(dtype);
}
#endif

void ComputePosePointToPlaneCPU(const float *source_points_ptr,
                                const float *target_points_ptr,
                                const float *target_normals_ptr,
                                const int64_t *correspondence_first,
                                const int64_t *correspondence_second,
                                const int n,
                                core::Tensor &pose,
                                const core::Dtype &dtype,
                                const core::Device &device) {
#ifdef _WIN32
    ComputePosePointToPlaneTBB(source_points_ptr, target_points_ptr,
                               target_normals_ptr, correspondence_first,
                               correspondence_second, n, pose, dtype, device);
#else
    ComputePosePointToPlaneOpenMP(source_points_ptr, target_points_ptr,
                                  target_normals_ptr, correspondence_first,
                                  correspondence_second, n, pose, dtype,
                                  device);
#endif
}

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
