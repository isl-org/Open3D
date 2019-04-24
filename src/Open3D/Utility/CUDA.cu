
#include "CUDA.cuh"

#include <iostream>
#include <sstream>
using namespace std;

// ----------------------------------------------------------------------------
// Diplay info about the specified device.
// Requires building the project in debug mode.
// ----------------------------------------------------------------------------
void open3d::cuda::DeviceInfo(const int& device_id) {
#ifndef NDEBUG
    if (-1 == device_id) return;

    cudaDeviceProp deviceProp;

    cudaGetDeviceProperties(&deviceProp, device_id);

    printf("GPU Device %d: %s, CUDA %d.%d\n", device_id, deviceProp.name,
           deviceProp.major, deviceProp.minor);
#endif
}

// ----------------------------------------------------------------------------
// Display debug info.
// Requires building the project in debug mode.
// ----------------------------------------------------------------------------
void open3d::cuda::DebugInfo(const string& function_name,
                             const cudaError_t& status) {
#ifndef NDEBUG
    if (cudaSuccess != status) {
        string error_message = cudaGetErrorString(status);
        printf("%20s: %s\n", function_name.c_str(), error_message.c_str());
    }
#endif
}
