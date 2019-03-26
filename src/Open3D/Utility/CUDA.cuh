
#pragma once

#include <cuda_runtime.h>

#include <string>

namespace open3d {
// set cuda device id to -1 in order to execute on the cpu
const int CPU = -1;

// Diplay info about the specified device.
std::string DeviceInfo(const int& device_id);
// Display debug info.
// Requires building the project in debug mode.
void DebugInfo(const std::string& function_name, const cudaError_t& status);

// Alocate device memory and perform validation.
template <typename T>
cudaError_t AllocateDeviceMemory(T** d,
                             const size_t& num_elements,
                             const int& device_id = 0) {
    cudaError_t status = cudaSuccess;

    if (CPU == device_id) return status;

    size_t num_bytes = num_elements * sizeof(T);

    cudaSetDevice(device_id);
    status = cudaMalloc((void**)d, num_bytes);
    DebugInfo("AllocateDeviceMemory", status);

    return status;
}

// Copy data to the device.
template <typename T>
cudaError_t CopyHst2DevMemory(T* h, T* d, const size_t& num_elements) {
    cudaError_t status = cudaSuccess;

    size_t num_bytes = num_elements * sizeof(T);

    status = cudaMemcpy(d, h, num_bytes, cudaMemcpyHostToDevice);

    DebugInfo("CopyHst2DevMemory", status);

    return status;
}

// Copy data from the device.
template <typename T>
cudaError_t CopyDev2HstMemory(T* d, T* h, const size_t& num_elements) {
    cudaError_t status = cudaSuccess;

    size_t num_bytes = num_elements * sizeof(T);

    status = cudaMemcpy(h, d, num_bytes, cudaMemcpyDeviceToHost);

    DebugInfo("CopyDev2HstMemory", status);

    return status;
}

// Safely deallocate device memory.
template <typename T>
cudaError_t ReleaseDeviceMemory(T** d) {
    cudaError_t status = cudaSuccess;

    status = cudaFree(*d);
    DebugInfo("ReleaseDeviceMemory", status);

    if (cudaSuccess == status) *d = NULL;

    return status;
}

// update the device memory on demand
template <typename T>
cudaError_t UpdateDeviceMemory(T** d_data,
                               const T* const data,
                               const size_t& num_elements,
                               const int& device_id = 0) {
    cudaError_t status = cudaSuccess;

    if (CPU == device_id) return status;

    if (*d_data != NULL) {
        status = cudaFree(*d_data);
        DebugInfo("UpdateDeviceMemory", status);
        if (cudaSuccess != status) return status;

        *d_data = NULL;
    }

    size_t num_bytes = num_elements * sizeof(T);

    cudaSetDevice(device_id);
    status = cudaMalloc((void**)d_data, num_bytes);
    DebugInfo("UpdateDeviceMemory", status);
    if (cudaSuccess != status) return status;

    status = cudaMemcpy(*d_data, data, num_bytes, cudaMemcpyHostToDevice);
    DebugInfo("UpdateDeviceMemory", status);
    if (cudaSuccess != status) return status;

    return status;
}

}  // namespace open3d
