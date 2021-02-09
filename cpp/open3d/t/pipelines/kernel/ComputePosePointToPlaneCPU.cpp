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

#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/CPULauncher.h"
#include "open3d/t/pipelines/kernel/ComputePosePointToPlaneImp.h"
#include "open3d/t/pipelines/kernel/TransformationConverter.h"
#include "open3d/utility/Timer.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {

// std::vector<double> Sum_(std::vector<double> &x,std::vector<double> &y){
//   std::vector<double> r(21,0);
//   int i;
//   for (i = 0; i < 21; i++)
//     r[i] = x[i] + y[i];

//   return r;
// }

void ComputePosePointToPlaneCPU(const float *src_pcd_ptr,
                                const float *tar_pcd_ptr,
                                const float *tar_norm_ptr,
                                const int64_t *corres_first,
                                const int64_t *corres_second,
                                const int n,
                                core::Tensor &pose,
                                const core::Dtype dtype,
                                const core::Device device) {
    utility::Timer time_reduction, time_kernel;
    time_kernel.Start();

    core::Tensor ATA =
            core::Tensor::Zeros({6, 6}, core::Dtype::Float64, device);
    core::Tensor ATA_ =
            core::Tensor::Zeros({1, 21}, core::Dtype::Float64, device);
    core::Tensor ATB =
            core::Tensor::Zeros({6, 1}, core::Dtype::Float64, device);

    double *ata_ptr = static_cast<double *>(ATA.GetDataPtr());
    double *ata_ = static_cast<double *>(ATA_.GetDataPtr());
    double *atb_ = static_cast<double *>(ATB.GetDataPtr());

#pragma omp parallel for reduction(+ : atb_[:6], ata_[:21])
    for (int64_t workload_idx = 0; workload_idx < n; ++workload_idx) {
        const int64_t &source_index = 3 * corres_first[workload_idx];
        const int64_t &target_index = 3 * corres_second[workload_idx];

        const float &sx = (src_pcd_ptr[source_index + 0]);
        const float &sy = (src_pcd_ptr[source_index + 1]);
        const float &sz = (src_pcd_ptr[source_index + 2]);
        const float &tx = (tar_pcd_ptr[target_index + 0]);
        const float &ty = (tar_pcd_ptr[target_index + 1]);
        const float &tz = (tar_pcd_ptr[target_index + 2]);
        const float &nx = (tar_norm_ptr[target_index + 0]);
        const float &ny = (tar_norm_ptr[target_index + 1]);
        const float &nz = (tar_norm_ptr[target_index + 2]);

        float ai[] = {(nz * sy - ny * sz),
                      (nx * sz - nz * sx),
                      (ny * sx - nx * sy),
                      nx,
                      ny,
                      nz};

        for (int i = 0, j = 0; j < 6; j++) {
            for (int k = 0; k <= j; k++) {
                // ATA_ {1,21}, as ATA {6,6} is a symmetric matrix.
                ata_[i] += ai[j] * ai[k];
                i++;
            }
            // ATB {6,1}.
            atb_[j] +=
                    ai[j] * ((tx - sx) * nx + (ty - sy) * ny + (tz - sz) * nz);
        }
    }

    // ATA_ {1,21} to ATA {6,6}.
    for (int i = 0, j = 0; j < 6; j++) {
        for (int k = 0; k <= j; k++) {
            ata_ptr[j * 6 + k] = ata_[i];
            ata_ptr[k * 6 + j] = ata_[i];
            i++;
        }
    }

    time_kernel.Stop();
    utility::LogInfo("         Kernel + Reduction: {}",
                     time_reduction.GetDuration());

    utility::Timer Solving_Pose_time_;
    Solving_Pose_time_.Start();
    // ATA(6,6) . Pose(6,1) = ATB(6,1)
    pose = ATA.Solve(ATB).Reshape({-1}).To(dtype);

    Solving_Pose_time_.Stop();
    utility::LogInfo("         Solving_Pose. Time: {}",
                     Solving_Pose_time_.GetDuration());
}

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
