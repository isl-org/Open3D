
#include "CUDA.cuh"

#include <iostream>
#include <sstream>
using namespace std;

// ----------------------------------------------------------------------------
// Diplay info about the specified device.
// ----------------------------------------------------------------------------
string open3d::DeviceInfo(const int& devID) {
    if (-1 == devID)
        return string();

    cudaDeviceProp deviceProp;

    cudaGetDeviceProperties(&deviceProp, devID);

    stringstream info;
    info << "GPU Device " << devID << ": ";
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

// ---------------------------------------------------------------------------
// Alocate device memory and perform validation.
// ---------------------------------------------------------------------------
cudaError_t open3d::AlocateDevMemory(double** d, const size_t& numElements,
    const int& devID) {
    cudaError_t status = cudaSuccess;

    if (CPU == devID) return status;

    size_t size = numElements * sizeof(double);

    cudaSetDevice(devID);
    status = cudaMalloc((void **)d, size);
    DebugInfo("AlocateDevMemory", status);

    return status;
}

// ---------------------------------------------------------------------------
// Copy data to the device.
// ---------------------------------------------------------------------------
cudaError_t open3d::CopyHst2DevMemory(double* h, double* d,
    const size_t& numElements) {
    cudaError_t status = cudaSuccess;

    size_t size = numElements * sizeof(double);

    status = cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);

    DebugInfo("CopyHst2DevMemory", status);

    return status;
}

// ---------------------------------------------------------------------------
// Copy data from the device.
// ---------------------------------------------------------------------------
cudaError_t open3d::CopyDev2HstMemory(double* d, double* h,
    const size_t& numElements) {
    cudaError_t status = cudaSuccess;

    size_t size = numElements * sizeof(double);

    status = cudaMemcpy(h, d, size, cudaMemcpyDeviceToHost);

    DebugInfo("CopyDev2HstMemory", status);

    return status;
}

// ---------------------------------------------------------------------------
// Safely deallocate device memory.
// ---------------------------------------------------------------------------
cudaError_t open3d::freeDev(double** d) {
    cudaError_t status = cudaSuccess;

    status = cudaFree(*d);
    DebugInfo("freeDev", status);

    if (cudaSuccess == status)
        *d = NULL;

    return status;
}

// ---------------------------------------------------------------------------
// update the device memory on demand
// ---------------------------------------------------------------------------
cudaError_t open3d::UpdateDeviceMemory(double **d_data,
    const double* const data,
    const size_t& numElements,
    const int& devID) {
    cudaError_t status = cudaSuccess;

    if (CPU == devID) return status;

    if (*d_data != NULL) {
        status = cudaFree(*d_data);
        DebugInfo("UpdateDeviceMemory", status);
        if (cudaSuccess != status) return status;

        *d_data = NULL;
    }

    size_t size = numElements * sizeof(double);

    cudaSetDevice(devID);
    status = cudaMalloc((void **)d_data, size);
    DebugInfo("UpdateDeviceMemory", status);
    if (cudaSuccess != status) return status;

    status = cudaMemcpy(*d_data, data, size, cudaMemcpyHostToDevice);
    DebugInfo("UpdateDeviceMemory", status);
    if (cudaSuccess != status) return status;

    return status;
}  // namespace geometry
