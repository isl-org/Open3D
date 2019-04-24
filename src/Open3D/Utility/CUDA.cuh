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

#pragma once

#include <cuda_runtime.h>
#include <string>

#include "Open3D/Types/DeviceID.h"

namespace open3d {
namespace cuda {

// Diplay info about the specified device.
void DeviceInfo(const int& device_id);
// Display debug info.
// Requires building the project in debug mode.
void DebugInfo(const std::string& function_name, const cudaError_t& status);

// Alocate device memory and perform validation.
template <typename T>
cudaError_t AllocateDeviceMemory(T** d_data,
                                 const size_t& num_elements,
                                 const DeviceID::Type& device_id) {
    cudaError_t status = cudaSuccess;

    int gpu_id = DeviceID::GPU_ID(device_id);

    // no GPU was selected
    if (gpu_id < 0) return status;

    size_t num_bytes = num_elements * sizeof(T);

    cudaSetDevice(gpu_id);
    status = cudaMalloc((void**)d_data, num_bytes);
    DebugInfo("AllocateDeviceMemory", status);

    cudaMemset(*d_data, 0, num_bytes);
    DebugInfo("AllocateDeviceMemory:init_zeros", status);

    return status;
}

// Copy data between the host and the device in any combination.
template <typename T>
cudaError_t Copy(const T* const src,
                 T* const dst,
                 const size_t& num_elements,
                 const cudaMemcpyKind& kind) {
    cudaError_t status = cudaSuccess;

    size_t num_bytes = num_elements * sizeof(T);

    status = cudaMemcpy(dst, src, num_bytes, kind);

    switch (kind) {
        case cudaMemcpyHostToHost:
            DebugInfo("cudaMemcpyHostToHost", status);
            break;
        case cudaMemcpyHostToDevice:
            DebugInfo("cudaMemcpyHostToDevice", status);
            break;
        case cudaMemcpyDeviceToHost:
            DebugInfo("cudaMemcpyDeviceToHost", status);
            break;
        case cudaMemcpyDeviceToDevice:
            DebugInfo("cudaMemcpyDeviceToDevice", status);
            break;
        case cudaMemcpyDefault:
            DebugInfo("cudaMemcpyDefault", status);
            break;
        default:
            break;
    }

    return status;
}

// Copy data from the host to the device.
template <typename T>
cudaError_t CopyHst2DevMemory(const T* const h_data,
                              T* const d_data,
                              const size_t& num_elements) {
    return Copy(h_data, d_data, num_elements, cudaMemcpyHostToDevice);
}

// Copy data from the device to the host.
template <typename T>
cudaError_t CopyDev2HstMemory(const T* const d_data,
                              T* const h_data,
                              const size_t& num_elements) {
    return Copy(d_data, h_data, num_elements, cudaMemcpyDeviceToHost);
}

// Copy data from the device to the device.
template <typename T>
cudaError_t CopyDev2DevMemory(const T* const d_data_src,
                              T* const d_data_dst,
                              const size_t& num_elements) {
    return Copy(d_data_src, d_data_dst, num_elements, cudaMemcpyDeviceToDevice);
}

// Safely deallocate device memory.
template <typename T>
cudaError_t ReleaseDeviceMemory(T** d_data) {
    cudaError_t status = cudaSuccess;

    if (NULL == *d_data) return status;

    status = cudaFree(*d_data);
    DebugInfo("ReleaseDeviceMemory", status);

    if (cudaSuccess == status) *d_data = NULL;

    return status;
}
}  // namespace cuda
}  // namespace open3d
