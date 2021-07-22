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
