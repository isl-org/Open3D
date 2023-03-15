// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cuda.h>
#include <cuda_runtime.h>

#include "open3d/t/pipelines/kernel/TransformationConverterImpl.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {

template <typename scalar_t>
__global__ void PoseToTransformationKernel(scalar_t *transformation_ptr,
                                           const scalar_t *X_ptr) {
    PoseToTransformationImpl(transformation_ptr, X_ptr);
}

template <typename scalar_t>
void PoseToTransformationCUDA(scalar_t *transformation_ptr,
                              const scalar_t *X_ptr) {
    utility::LogError("Unsupported data type.");
}

template <>
void PoseToTransformationCUDA<float>(float *transformation_ptr,
                                     const float *X_ptr) {
    PoseToTransformationKernel<float>
            <<<1, 1, 0, core::cuda::GetStream()>>>(transformation_ptr, X_ptr);
}

template <>
void PoseToTransformationCUDA<double>(double *transformation_ptr,
                                      const double *X_ptr) {
    PoseToTransformationKernel<double>
            <<<1, 1, 0, core::cuda::GetStream()>>>(transformation_ptr, X_ptr);
}

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
