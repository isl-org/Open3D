// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Tensor.h"
namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {

void FillInRigidAlignmentTerm(core::Tensor &AtA,
                              core::Tensor &Atb,
                              core::Tensor &residual,
                              const core::Tensor &Ti_qs,
                              const core::Tensor &Tj_qs,
                              const core::Tensor &Ri_normal_ps,
                              int i,
                              int j,
                              float threshold);

void FillInSLACAlignmentTerm(core::Tensor &AtA,
                             core::Tensor &Atb,
                             core::Tensor &residual,
                             const core::Tensor &Ti_qs,
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
                             float threshold);

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
                               int anchor_idx);

void FillInRigidAlignmentTermCPU(core::Tensor &AtA,
                                 core::Tensor &Atb,
                                 core::Tensor &residual,
                                 const core::Tensor &Ti_qs,
                                 const core::Tensor &Tj_qs,
                                 const core::Tensor &Ri_normal_ps,
                                 int i,
                                 int j,
                                 float threshold);

void FillInSLACAlignmentTermCPU(core::Tensor &AtA,
                                core::Tensor &Atb,
                                core::Tensor &residual,
                                const core::Tensor &Ti_qs,
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
                                float threshold);

void FillInSLACRegularizerTermCPU(core::Tensor &AtA,
                                  core::Tensor &Atb,
                                  core::Tensor &residual,
                                  const core::Tensor &grid_idx,
                                  const core::Tensor &grid_nbs_idx,
                                  const core::Tensor &grid_nbs_mask,
                                  const core::Tensor &positions_init,
                                  const core::Tensor &positions_curr,
                                  float weight,
                                  int n,
                                  int anchor_idx);

#ifdef BUILD_CUDA_MODULE
void FillInRigidAlignmentTermCUDA(core::Tensor &AtA,
                                  core::Tensor &Atb,
                                  core::Tensor &residual,
                                  const core::Tensor &Ti_qs,
                                  const core::Tensor &Tj_qs,
                                  const core::Tensor &Ri_normal_ps,
                                  int i,
                                  int j,
                                  float threshold);

void FillInSLACAlignmentTermCUDA(core::Tensor &AtA,
                                 core::Tensor &Atb,
                                 core::Tensor &residual,
                                 const core::Tensor &Ti_qs,
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
                                 float threshold);

void FillInSLACRegularizerTermCUDA(core::Tensor &AtA,
                                   core::Tensor &Atb,
                                   core::Tensor &residual,
                                   const core::Tensor &grid_idx,
                                   const core::Tensor &grid_nbs_idx,
                                   const core::Tensor &grid_nbs_mask,
                                   const core::Tensor &positions_init,
                                   const core::Tensor &positions_curr,
                                   float weight,
                                   int n,
                                   int anchor_idx);

#endif

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
