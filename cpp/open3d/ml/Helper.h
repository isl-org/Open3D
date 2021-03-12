// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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

/// \file Helper.h
/// \brief Helper functions for the ml ops

#pragma once

#ifdef BUILD_CUDA_MODULE

#include <cuda.h>
#include <cuda_runtime.h>

#endif  // #ifdef BUILD_CUDA_MODULE

#include <stdio.h>

#include <stdexcept>
#include <string>

#ifdef BUILD_CUDA_MODULE
/// TODO: Link Open3D and use OPEN3D_CUDA_CHECK instead.
#define OPEN3D_ML_CUDA_CHECK(err) \
    { open3d::ml::__OPEN3D_ML_CUDA_CHECK((err), __FILE__, __LINE__); }
#endif

namespace open3d {
namespace ml {

#ifdef BUILD_CUDA_MODULE
/// Returns the texture alignment in bytes for the current device.
inline int GetCUDACurrentDeviceTextureAlignment() {
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        throw std::runtime_error(
                "GetCUDACurrentDeviceTextureAlignment(): cudaGetDevice failed "
                "with {}" +
                std::string(cudaGetErrorString(err)));
    }

    int value = 0;
    err = cudaDeviceGetAttribute(&value, cudaDevAttrTextureAlignment, device);
    if (err != cudaSuccess) {
        throw std::runtime_error(
                "GetCUDACurrentDeviceTextureAlignment(): cudaGetDevice failed "
                "with {}" +
                std::string(cudaGetErrorString(err)));
    }
    return value;
}

/// TODO: Link Open3D and use OPEN3D_CUDA_CHECK instead.
inline void __OPEN3D_ML_CUDA_CHECK(cudaError_t err,
                                   const char *file,
                                   int line,
                                   bool abort = true) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s:%d CUDA runtime error: %s\n", file, line,
                cudaGetErrorString(err));
        if (abort) {
            exit(err);
        }
    }
}
#endif

}  // namespace ml
}  // namespace open3d
