
#pragma once

#include <cuda_runtime.h>

#include <string>

namespace open3d {
    // Diplay info about the specified device.
    void DeviceInfo(const int& devID);
    // Alocate host memory and perform validation.
    bool AlocateHstMemory(double** h, const int& numElements, const std::string& name);
    // Alocate device memory and perform validation.
    bool AlocateDevMemory(double** d, const int& numElements, const std::string& name);
    // Initialize host memory.
    void RandInit(double* h, const int& numElements);
    // Copy data to the device.
    bool CopyHst2DevMemory(double* h, double* d, const int& numElements);
    // Copy data from the device.
    bool CopyDev2HstMemory(double* d, double* h, const int& numElements);
    // Safely deallocate device memory.
    bool freeDev(double** d, const std::string& name);
    // update the device memory on demand
    bool UpdateDeviceMemory(double **d_data,
        const double* const data,
        const size_t& size);
}