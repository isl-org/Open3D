
#include "CUDA.cuh"

#include <iostream>
#include <sstream>
using namespace std;

// ----------------------------------------------------------------------------
// Diplay info about the specified device.
// ----------------------------------------------------------------------------
string open3d::DeviceInfo(const int& device_id) {
    if (-1 == device_id)
        return string();

    cudaDeviceProp deviceProp;

    cudaGetDeviceProperties(&deviceProp, device_id);

    stringstream info;
    info << "GPU Device " << device_id << ": ";
    info << deviceProp.name << ", ";
    info << "CUDA ";
    info << deviceProp.major << ".";
    info << deviceProp.minor << endl;

    return info.str();
}

// ----------------------------------------------------------------------------
// Display debug info.
// Requires building the project in debug mode.
// ----------------------------------------------------------------------------
void open3d::DebugInfo(const string& function_name, const cudaError_t& status) {
    #ifndef NDEBUG
    if (cudaSuccess != status) {
        string error_message = cudaGetErrorString(status);
        printf("%20s: %s\n", function_name.c_str(), error_message.c_str());
    }
    #endif
}
