
#pragma once

#include <cuda_runtime.h>

#include <string>

namespace open3d {
    // set cuda device id to -1 in order to execute on the cpu
    const int CPU = -1;

    // Diplay info about the specified device.
    std::string DeviceInfo(const int& devID);
    // Display debug info.
    // Requires building the project in debug mode.
    void DebugInfo(const std::string& function_name, const cudaError_t& status);
    // Alocate device memory and perform validation.
    cudaError_t AlocateDevMemory(double** d, const size_t& numElements,
        const int& devID = 0);
    // Copy data to the device.
    cudaError_t CopyHst2DevMemory(double* h, double* d,
        const size_t& numElements);
    // Copy data from the device.
    cudaError_t CopyDev2HstMemory(double* d, double* h,
        const size_t& numElements);
    // Safely deallocate device memory.
    cudaError_t freeDev(double** d);
    // update the device memory on demand
    cudaError_t UpdateDeviceMemory(double **d_data, const double* const data,
        const size_t& numElements, const int& devID = 0);
}
