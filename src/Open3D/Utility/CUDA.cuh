
#pragma once

#include <cuda_runtime.h>

#include <string>

namespace open3d {
    // Diplay info about the specified device.
    std::string DeviceInfo(const int& devID);
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
