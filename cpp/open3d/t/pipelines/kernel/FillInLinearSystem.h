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
