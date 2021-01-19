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
#include "open3d/core/kernel/CUDALauncher.cuh"
#include "open3d/t/pipelines/kernel/ComputeTransformPointToPlane.h"
#include "open3d/t/pipelines/kernel/TransformationConverter.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {

void ComputeTransformPointToPlaneCUDA(const float *src_pcd_ptr,
                                      const float *tar_pcd_ptr,
                                      const float *tar_norm_ptr,
                                      const int n,
                                      core::Tensor &tranformation,
                                      const core::Dtype dtype,
                                      const core::Device device) {
    core::Tensor atai = core::Tensor::Empty({n, 21}, dtype, device);
    core::Tensor atbi = core::Tensor::Empty({n, 6}, dtype, device);
    float *atai_ptr = static_cast<float *>(atai.GetDataPtr());
    float *atbi_ptr = static_cast<float *>(atbi.GetDataPtr());

    core::kernel::CUDALauncher::LaunchGeneralKernel(
            n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
                const int64_t pcd_stride = 3 * workload_idx;
                const int64_t atai_stride = 21 * workload_idx;
                const int64_t atbi_stride = 6 * workload_idx;

                const float sx = src_pcd_ptr[pcd_stride + 0];
                const float sy = src_pcd_ptr[pcd_stride + 1];
                const float sz = src_pcd_ptr[pcd_stride + 2];
                const float tx = tar_pcd_ptr[pcd_stride + 0];
                const float ty = tar_pcd_ptr[pcd_stride + 1];
                const float tz = tar_pcd_ptr[pcd_stride + 2];
                const float nx = tar_norm_ptr[pcd_stride + 0];
                const float ny = tar_norm_ptr[pcd_stride + 1];
                const float nz = tar_norm_ptr[pcd_stride + 2];

                float bi = (tx - sx) * nx + (ty - sy) * ny + (tz - sz) * nz;
                float ai[] = {(nz * sy - ny * sz),
                              (nx * sz - nz * sx),
                              (ny * sx - nx * sy),
                              nx,
                              ny,
                              ny};

                for (int i = 0, j = 0; j < 6; j++) {
                    for (int k = j; k < 6; k++) {
                        atai_ptr[atai_stride + i] = ai[j] * ai[k];
                        i++;
                    }
                    atbi_ptr[atbi_stride + j] = ai[j] * bi;
                }
            });

    /*  // Solve on CUDA

        // Reduce matrix atai (to 1x21) and atbi (to 1x6)
        core::Tensor ata_1x21 = atai.Sum({0}, true);
        core::Tensor atb_t = atbi.Sum({0}, true);

        // Get the ATA matrix back:
        // Getting ATA and ATB in orginial form is NOT required if LeastSq is
       going
        // to be hardcoded with values.
        core::Tensor ATA = core::Tensor::Zeros({6, 6}, dtype, device);
        float *ATA_ptr = static_cast<float *>(ATA.GetDataPtr());
        const float *ata_1x21_ptr =
                static_cast<const float *>(ata_1x21.GetDataPtr());

        core::kernel::CUDALauncher::LaunchGeneralKernel(
                1, [=] OPEN3D_DEVICE(int64_t workload_idx) {
                    for (int i = 0, j = 0; j < 6; j++) {
                        for (int k = j; k < 6; k++) {
                            ATA_ptr[j * 6 + k] = ata_1x21_ptr[i];
                            ATA_ptr[k * 6 + j] = ata_1x21_ptr[i];
                            i++;
                        }
                    }
                });

        core::Tensor ATB = atb_t.T();
        utility::LogInfo(" ATA: \n{},\n ATB: \n{}", ATA.ToString(),
       ATB.ToString()); core::Tensor Pose =
       (ATA.LeastSquares(ATB)).Reshape({-1}); utility::LogInfo("\n POSE: \n{}",
       Pose.ToString()); tranformation =
       t::pipelines::kernel::PoseToTransformation(Pose);
    */

    // Solve on CPU
    core::Device solve_device = core::Device("CPU:0");
    // Reduce matrix atai (to 1x21) and atbi (to 1x6)
    core::Tensor ata_1x21 = atai.Sum({0}, true).To(solve_device);
    core::Tensor atb_t = atbi.Sum({0}, true).To(solve_device);

    // Get the ATA matrix back:
    // Getting ATA and ATB in orginial form is NOT required if LeastSq is going
    // to be hardcoded with values.
    core::Tensor ATA = core::Tensor::Zeros({6, 6}, dtype, solve_device);
    float *ATA_ptr = static_cast<float *>(ATA.GetDataPtr());
    const float *ata_1x21_ptr =
            static_cast<const float *>(ata_1x21.GetDataPtr());

    for (int i = 0, j = 0; j < 6; j++) {
        for (int k = j; k < 6; k++) {
            ATA_ptr[j * 6 + k] = ata_1x21_ptr[i];
            ATA_ptr[k * 6 + j] = ata_1x21_ptr[i];
            i++;
        }
    }

    core::Tensor ATB = atb_t.T();
    utility::LogInfo(" ATA: \n{},\n ATB: \n{}", ATA.ToString(), ATB.ToString());
    core::Tensor Pose = (ATA.LeastSquares(ATB)).Reshape({-1});
    utility::LogInfo("\n POSE: \n{}", Pose.ToString());
    tranformation = t::pipelines::kernel::PoseToTransformation(Pose).To(device);

    return;
}

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
