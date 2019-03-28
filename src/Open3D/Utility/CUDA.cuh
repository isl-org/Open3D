
#pragma once

#include <cuda_runtime.h>

#include <string>

namespace open3d {
    namespace cuda {
// This enum is used for managing memory and execution for multiple devices.
// At them moment, it supports 1xCPU and up to 8xGPUs.
// We can have at most 1xCPU and 1xGPU simultaneously.
namespace DeviceID {
enum Type {
    GPU_00 = 1 << 0,
    GPU_01 = 1 << 1,
    GPU_02 = 1 << 2,
    GPU_03 = 1 << 3,
    GPU_04 = 1 << 4,
    GPU_05 = 1 << 5,
    GPU_06 = 1 << 6,
    GPU_07 = 1 << 7,
    CPU = 1 << 8
};

inline int GPU_ID(const DeviceID::Type& device_id) {
    // if present, a GPU id must be greater than zero
    // a negative value means no GPU was selected
    int gpu_id = -1;

    if (DeviceID::GPU_00 & device_id) gpu_id = 0;
    if (DeviceID::GPU_01 & device_id) gpu_id = 1;
    if (DeviceID::GPU_02 & device_id) gpu_id = 2;
    if (DeviceID::GPU_03 & device_id) gpu_id = 3;
    if (DeviceID::GPU_04 & device_id) gpu_id = 4;
    if (DeviceID::GPU_05 & device_id) gpu_id = 5;
    if (DeviceID::GPU_06 & device_id) gpu_id = 6;
    if (DeviceID::GPU_07 & device_id) gpu_id = 7;

    return gpu_id;
}
}  // namespace DeviceID

// Diplay info about the specified device.
std::string DeviceInfo(const int& device_id);
// Display debug info.
// Requires building the project in debug mode.
void DebugInfo(const std::string& function_name, const cudaError_t& status);

// Alocate device memory and perform validation.
template <typename T>
cudaError_t AllocateDeviceMemory(T** d_data,
                                 const size_t& num_bytes,
                                 const DeviceID::Type& device_id) {
    cudaError_t status = cudaSuccess;

    int gpu_id = DeviceID::GPU_ID(device_id);

    // no GPU was selected
    if (gpu_id < 0) return status;

    cudaSetDevice(gpu_id);
    status = cudaMalloc((void**)d_data, num_bytes);
    DebugInfo("AllocateDeviceMemory", status);

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

    switch (kind)
    {
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
    cudaError_t status = cudaSuccess;

    size_t num_bytes = num_elements * sizeof(T);

    status = cudaMemcpy(d_data, h_data, num_bytes, cudaMemcpyHostToDevice);

    DebugInfo("CopyHst2DevMemory", status);

    return status;
}

// Copy data from the device to the host.
template <typename T>
cudaError_t CopyDev2HstMemory(const T* const d_data,
                              T* const h_data,
                              const size_t& num_elements) {
    cudaError_t status = cudaSuccess;

    size_t num_bytes = num_elements * sizeof(T);

    status = cudaMemcpy(h_data, d_data, num_bytes, cudaMemcpyDeviceToHost);

    DebugInfo("CopyDev2HstMemory", status);

    return status;
}

// Copy data from the device to the device.
template <typename T>
cudaError_t CopyDev2DevMemory(const T* const d_data_src,
                              T* const d_data_dst,
                              const size_t& num_elements) {
    cudaError_t status = cudaSuccess;

    size_t num_bytes = num_elements * sizeof(T);

    status = cudaMemcpy(d_data_dst, d_data_src, num_bytes,
                        cudaMemcpyDeviceToDevice);

    DebugInfo("CopyDev2DevMemory", status);

    return status;
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
    } // namespace cuda
}  // namespace open3d
