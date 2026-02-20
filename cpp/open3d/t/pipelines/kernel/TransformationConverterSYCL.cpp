// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/Device.h"
#include "open3d/core/Dispatch.h"
#include "open3d/core/SYCLContext.h"
#include "open3d/t/pipelines/kernel/TransformationConverterImpl.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {

template <typename scalar_t>
void PoseToTransformationSYCL(scalar_t *transformation_ptr,
                               const scalar_t *pose_ptr,
                               const core::Device &device) {
    sycl::queue queue =
            core::sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    queue.single_task([=]() {
             PoseToTransformationImpl(transformation_ptr, pose_ptr);
         }).wait_and_throw();
}

template <typename scalar_t>
void TransformationToPoseSYCL(scalar_t *pose_ptr,
                               const scalar_t *transformation_ptr,
                               const core::Device &device) {
    sycl::queue queue =
            core::sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    queue.single_task([=]() {
             TransformationToPoseImpl(pose_ptr, transformation_ptr);
         }).wait_and_throw();
}

template void PoseToTransformationSYCL<float>(float *,
                                              const float *,
                                              const core::Device &);
template void PoseToTransformationSYCL<double>(double *,
                                               const double *,
                                               const core::Device &);
template void TransformationToPoseSYCL<float>(float *,
                                              const float *,
                                              const core::Device &);
template void TransformationToPoseSYCL<double>(double *,
                                               const double *,
                                               const core::Device &);

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
