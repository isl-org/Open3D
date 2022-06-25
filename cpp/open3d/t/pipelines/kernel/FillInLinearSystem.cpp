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

#include "open3d/t/pipelines/kernel/FillInLinearSystem.h"

#include "open3d/core/TensorCheck.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {

void FillInRigidAlignmentTerm(core::Tensor &AtA,
                              core::Tensor &Atb,
                              core::Tensor &residual,
                              const core::Tensor &Ti_ps,
                              const core::Tensor &Tj_qs,
                              const core::Tensor &Ri_normal_ps,
                              int i,
                              int j,
                              float threshold) {
    core::AssertTensorDtype(AtA, core::Float32);
    core::AssertTensorDtype(Atb, core::Float32);
    core::AssertTensorDtype(residual, core::Float32);
    core::AssertTensorDtype(Ti_ps, core::Float32);
    core::AssertTensorDtype(Tj_qs, core::Float32);
    core::AssertTensorDtype(Ri_normal_ps, core::Float32);

    core::Device device = AtA.GetDevice();
    if (Atb.GetDevice() != device) {
        utility::LogError("AtA should have the same device as Atb.");
    }
    if (Ti_ps.GetDevice() != device) {
        utility::LogError(
                "Points i should have the same device as the linear system.");
    }
    if (Tj_qs.GetDevice() != device) {
        utility::LogError(
                "Points j should have the same device as the linear system.");
    }
    if (Ri_normal_ps.GetDevice() != device) {
        utility::LogError(
                "Normals i should have the same device as the linear system.");
    }

    if (AtA.IsCPU()) {
        FillInRigidAlignmentTermCPU(AtA, Atb, residual, Ti_ps, Tj_qs,
                                    Ri_normal_ps, i, j, threshold);

    } else if (AtA.IsCUDA()) {
#ifdef BUILD_CUDA_MODULE
        FillInRigidAlignmentTermCUDA(AtA, Atb, residual, Ti_ps, Tj_qs,
                                     Ri_normal_ps, i, j, threshold);

#else
        utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
    } else {
        utility::LogError("Unimplemented device");
    }
}

void FillInSLACAlignmentTerm(core::Tensor &AtA,
                             core::Tensor &Atb,
                             core::Tensor &residual,
                             const core::Tensor &Ti_ps,
                             const core::Tensor &Tj_qs,
                             const core::Tensor &normal_ps,
                             const core::Tensor &Ri_normal_ps,
                             const core::Tensor &RjT_Ri_normal_ps,
                             const core::Tensor &cgrid_idx_ps,
                             const core::Tensor &cgrid_idx_qs,
                             const core::Tensor &cgrid_ratio_qs,
                             const core::Tensor &cgrid_ratio_ps,
                             int i,
                             int j,
                             int n,
                             float threshold) {
    core::AssertTensorDtype(AtA, core::Float32);
    core::AssertTensorDtype(Atb, core::Float32);
    core::AssertTensorDtype(residual, core::Float32);
    core::AssertTensorDtype(Ti_ps, core::Float32);
    core::AssertTensorDtype(Tj_qs, core::Float32);
    core::AssertTensorDtype(normal_ps, core::Float32);
    core::AssertTensorDtype(Ri_normal_ps, core::Float32);
    core::AssertTensorDtype(RjT_Ri_normal_ps, core::Float32);

    core::Device device = AtA.GetDevice();
    if (Atb.GetDevice() != device) {
        utility::LogError("AtA should have the same device as Atb.");
    }
    if (Ti_ps.GetDevice() != device) {
        utility::LogError(
                "Points i should have the same device as the linear system.");
    }
    if (Tj_qs.GetDevice() != device) {
        utility::LogError(
                "Points j should have the same device as the linear system.");
    }
    if (Ri_normal_ps.GetDevice() != device) {
        utility::LogError(
                "Normals i should have the same device as the linear system.");
    }

    if (AtA.IsCPU()) {
        FillInSLACAlignmentTermCPU(AtA, Atb, residual, Ti_ps, Tj_qs, normal_ps,
                                   Ri_normal_ps, RjT_Ri_normal_ps, cgrid_idx_ps,
                                   cgrid_idx_qs, cgrid_ratio_ps, cgrid_ratio_qs,
                                   i, j, n, threshold);

    } else if (AtA.IsCUDA()) {
#ifdef BUILD_CUDA_MODULE
        FillInSLACAlignmentTermCUDA(AtA, Atb, residual, Ti_ps, Tj_qs, normal_ps,
                                    Ri_normal_ps, RjT_Ri_normal_ps,
                                    cgrid_idx_ps, cgrid_idx_qs, cgrid_ratio_ps,
                                    cgrid_ratio_qs, i, j, n, threshold);

#else
        utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
    } else {
        utility::LogError("Unimplemented device");
    }
}

void FillInSLACRegularizerTerm(core::Tensor &AtA,
                               core::Tensor &Atb,
                               core::Tensor &residual,
                               const core::Tensor &grid_idx,
                               const core::Tensor &grid_nbs_idx,
                               const core::Tensor &grid_nbs_mask,
                               const core::Tensor &positions_init,
                               const core::Tensor &positions_curr,
                               float weight,
                               int n,
                               int anchor_idx) {
    core::AssertTensorDtype(AtA, core::Float32);
    core::AssertTensorDtype(Atb, core::Float32);
    core::AssertTensorDtype(residual, core::Float32);

    core::Device device = AtA.GetDevice();
    if (Atb.GetDevice() != device) {
        utility::LogError("AtA should have the same device as Atb.");
    }

    if (AtA.IsCPU()) {
        FillInSLACRegularizerTermCPU(AtA, Atb, residual, grid_idx, grid_nbs_idx,
                                     grid_nbs_mask, positions_init,
                                     positions_curr, weight, n, anchor_idx);

    } else if (AtA.IsCUDA()) {
#ifdef BUILD_CUDA_MODULE
        FillInSLACRegularizerTermCUDA(
                AtA, Atb, residual, grid_idx, grid_nbs_idx, grid_nbs_mask,
                positions_init, positions_curr, weight, n, anchor_idx);
#else
        utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
    } else {
        utility::LogError("Unimplemented device");
    }
}

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
