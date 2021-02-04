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

// Private header. Do not include in Open3d.h.

#pragma once

#include "open3d/core/Tensor.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {

void ComputePosePointToPlaneCPU(const float *src_pcd_ptr,
                                const float *tar_pcd_ptr,
                                const float *tar_norm_ptr,
                                const int n,
                                core::Tensor &pose,
                                const core::Dtype dtype,
                                const core::Device device);

#ifdef BUILD_CUDA_MODULE
void ComputePosePointToPlaneCUDA(const float *src_pcd_ptr,
                                 const float *tar_pcd_ptr,
                                 const float *tar_norm_ptr,
                                 const int n,
                                 core::Tensor &pose,
                                 const core::Dtype dtype,
                                 const core::Device device);
#endif

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
