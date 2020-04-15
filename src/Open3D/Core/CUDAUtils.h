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

/// \file CUDAUtils.h
/// \brief Common CUDA utilities
///
/// CUDAUtils.h may be included from CPU-only code.
/// Use #ifdef __CUDACC__ to mark conitional compilation

#pragma once

#include "Open3D/Utility/Console.h"

#ifdef BUILD_CUDA_MODULE

#include <cuda.h>
#include <cuda_runtime.h>

#define OPEN3D_HOST_DEVICE __host__ __device__
#define OPEN3D_DEVICE __device__
#define OPEN3D_ASSERT_HOST_DEVICE_LAMBDA(type)                            \
    static_assert(__nv_is_extended_host_device_lambda_closure_type(type), \
                  #type " must be a __host__ __device__ lambda")
#define OPEN3D_CUDA_CHECK(err) \
    open3d::__OPEN3D_CUDA_CHECK(err, __FILE__, __LINE__)
#define OPEN3D_GET_LAST_CUDA_ERROR(message) \
    __OPEN3D_GET_LAST_CUDA_ERROR(message, __FILE__, __LINE__)

#else  // #ifdef BUILD_CUDA_MODULE

#define OPEN3D_HOST_DEVICE
#define OPEN3D_DEVICE
#define OPEN3D_ASSERT_HOST_DEVICE_LAMBDA(type)
#define OPEN3D_CUDA_CHECK(err)
#define OPEN3D_GET_LAST_CUDA_ERROR(message)

#endif  // #ifdef BUILD_CUDA_MODULE

namespace open3d {

#ifdef BUILD_CUDA_MODULE
inline void __OPEN3D_CUDA_CHECK(cudaError_t err,
                                const char* file,
                                const int line) {
    if (err != cudaSuccess) {
        utility::LogError("{}:{} CUDA runtime error: {}", file, line,
                          cudaGetErrorString(err));
    }
}

inline void __OPEN3D_GET_LAST_CUDA_ERROR(const char* message,
                                         const char* file,
                                         const int line) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        utility::LogError("{}:{} {}: OPEN3D_GET_LAST_CUDA_ERROR(): {}", file,
                          line, message, cudaGetErrorString(err));
    }
}
#endif

namespace cuda {

int DeviceCount();
bool IsAvailable();

}  // namespace cuda
}  // namespace open3d
