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

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>

#include "open3d/t/pipelines/SolveTransformation.h"

namespace open3d {
namespace t {
namespace pipelines {

__global__ void ComputeTransformationFromPoseCUDAKernel(
        float *X_ptr, float *transformation_ptr) {
    // Kernel launched with only 1 thread
    // Rotation from Pose X
    transformation_ptr[0] = std::cos(X_ptr[2]) * std::cos(X_ptr[1]);
    transformation_ptr[1] =
            -1 * std::sin(X_ptr[2]) * std::cos(X_ptr[0]) +
            std::cos(X_ptr[2]) * std::sin(X_ptr[1]) * std::sin(X_ptr[0]);
    transformation_ptr[2] =
            std::sin(X_ptr[2]) * std::sin(X_ptr[0]) +
            std::cos(X_ptr[2]) * std::sin(X_ptr[1]) * std::cos(X_ptr[0]);
    transformation_ptr[4] = std::sin(X_ptr[2]) * std::cos(X_ptr[1]);
    transformation_ptr[5] =
            std::cos(X_ptr[2]) * std::cos(X_ptr[0]) +
            std::sin(X_ptr[2]) * std::sin(X_ptr[1]) * std::sin(X_ptr[0]);
    transformation_ptr[6] =
            -1 * std::cos(X_ptr[2]) * std::sin(X_ptr[0]) +
            std::sin(X_ptr[2]) * std::sin(X_ptr[1]) * std::cos(X_ptr[0]);
    transformation_ptr[8] = -1 * std::sin(X_ptr[1]);
    transformation_ptr[9] = std::cos(X_ptr[1]) * std::sin(X_ptr[0]);
    transformation_ptr[10] = std::cos(X_ptr[1]) * std::cos(X_ptr[0]);
}

core::Tensor ComputeTransformationFromPoseCUDA(const core::Tensor &X) {
    core::Dtype dtype = core::Dtype::Float32;
    core::Device device = X.GetDevice();
    core::Tensor transformation = core::Tensor::Zeros({4, 4}, dtype, device);
    transformation = transformation.Contiguous();
    auto X_copy = X.Contiguous();
    float *transformation_ptr =
            static_cast<float *>(transformation.GetDataPtr());
    float *X_ptr = static_cast<float *>(X_copy.GetDataPtr());

    // kernel call
    ComputeTransformationFromPoseCUDAKernel<<<1, 1>>>(X_ptr,
                                                      transformation_ptr);

    // Translation from Pose X
    transformation.SetItem(
            {core::TensorKey::Slice(0, 3, 1), core::TensorKey::Slice(3, 4, 1)},
            X.GetItem({core::TensorKey::Slice(3, 6, 1)}).Reshape({3, 1}));
    // Current Implementation DOES NOT SUPPORT SCALE transfomation
    transformation[3][3] = 1;
    return transformation;
}

}  // namespace pipelines
}  // namespace t
}  // namespace open3d
