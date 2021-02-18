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

void ComputePosePointToPlaneCPU(const float *src_pcd_ptr,
                                const float *tar_pcd_ptr,
                                const float *tar_norm_ptr,
                                const int64_t *corres_first,
                                const int64_t *corres_second,
                                const int n,
                                core::Tensor &pose,
                                const core::Dtype dtype,
                                const core::Device device) {
    core::Tensor ATA =
            core::Tensor::Zeros({6, 6}, core::Dtype::Float64, device);
    core::Tensor ATA_1x21 =
            core::Tensor::Zeros({1, 21}, core::Dtype::Float64, device);
    core::Tensor ATB =
            core::Tensor::Zeros({6, 1}, core::Dtype::Float64, device);

    double *ata_ptr = static_cast<double *>(ATA.GetDataPtr());
    double *ata_1x21 = static_cast<double *>(ATA_1x21.GetDataPtr());
    double *atb_ptr = static_cast<double *>(ATB.GetDataPtr());

#pragma omp parallel for reduction(+ : atb_ptr[:6], ata_1x21[:21])
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
                ata_1x21[i] += ai[j] * ai[k];
                i++;
            }
            // ATB {6,1}.
            atb_ptr[j] +=
                    ai[j] * ((tx - sx) * nx + (ty - sy) * ny + (tz - sz) * nz);
        }
    }

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

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
