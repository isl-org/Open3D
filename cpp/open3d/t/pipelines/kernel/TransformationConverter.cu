// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
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

template <typename scalar_t>
__global__ void TransformationToPoseKernel(scalar_t *X_ptr,
                                           const scalar_t *transformation_ptr) {
    TransformationToPoseImpl(X_ptr, transformation_ptr);
}

template <typename scalar_t>
void TransformationToPoseCUDA(scalar_t *X_ptr,
                              const scalar_t *transformation_ptr) {
    utility::LogError("Unsupported data type.");
}

template <>
void TransformationToPoseCUDA<float>(float *X_ptr,
                                     const float *transformation_ptr) {
    TransformationToPoseKernel<float>
            <<<1, 1, 0, core::cuda::GetStream()>>>(X_ptr, transformation_ptr);
}

template <>
void TransformationToPoseCUDA<double>(double *X_ptr,
                                      const double *transformation_ptr) {
    TransformationToPoseKernel<double>
            <<<1, 1, 0, core::cuda::GetStream()>>>(X_ptr, transformation_ptr);
}

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
