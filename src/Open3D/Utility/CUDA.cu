
#include "CUDA.cuh"

#include <iostream>
using namespace std;

// ----------------------------------------------------------------------------
// Diplay info about the specified device.
// ----------------------------------------------------------------------------
void open3d::DeviceInfo(const int& devID)
{
    cudaDeviceProp deviceProp;

    cudaGetDeviceProperties(&deviceProp, devID);

    cout << "GPU Device " << devID << ": ";
    cout << deviceProp.name << ", ";
    cout << "CUDA ";
    cout << deviceProp.major << ".";
    cout << deviceProp.minor << endl;
    cout << endl;
}

// ---------------------------------------------------------------------------
// Alocate host memory and perform validation.
// ---------------------------------------------------------------------------
bool open3d::AlocateHstMemory(double** h, const int& numElements, const string& name)
{
    size_t size = numElements * sizeof(double);

    *h = (double *)malloc(size);

    if (*h != NULL)
        return true;

    cout << "Failed to allocate host memory: " << name << endl;

    return false;
}

// ---------------------------------------------------------------------------
// Alocate device memory and perform validation.
// ---------------------------------------------------------------------------
bool open3d::AlocateDevMemory(double** d, const int& numElements, const string& name)
{
    cudaError_t status = cudaSuccess;

    size_t size = numElements * sizeof(double);

    status = cudaMalloc((void **)d, size);

    if (status == cudaSuccess)
        return true;

    cout << "status: " << cudaGetErrorString(status) << endl;
    cout << "Failed to allocate device memory: " << name << endl;

    return false;
}

// ---------------------------------------------------------------------------
// Initialize host memory.
// ---------------------------------------------------------------------------
void open3d::RandInit(double* h, const int& numElements)
{
    for (int i = 0; i < numElements; ++i)
        h[i] = rand()/(double)RAND_MAX;
}

// ---------------------------------------------------------------------------
// Copy data to the device.
// ---------------------------------------------------------------------------
bool open3d::CopyHst2DevMemory(double* h, double* d, const int& numElements)
{
    cudaError_t status = cudaSuccess;

    size_t size = numElements * sizeof(double);

    status = cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);

    if (status == cudaSuccess)
        return true;

    cout << "status: " << cudaGetErrorString(status) << endl;
    cout << "Failed to copy host memory to the CUDA device." << endl;

    return false;
}

// ---------------------------------------------------------------------------
// Copy data from the device.
// ---------------------------------------------------------------------------
bool open3d::CopyDev2HstMemory(double* d, double* h, const int& numElements)
{
    cudaError_t status = cudaSuccess;

    size_t size = numElements * sizeof(double);

    status = cudaMemcpy(h, d, size, cudaMemcpyDeviceToHost);

    if (status == cudaSuccess)
        return true;

    cout << "status: " << cudaGetErrorString(status) << endl;
    cout << "Failed to copy device memory to the host." << endl;

    return false;
}

// ---------------------------------------------------------------------------
// Safely deallocate device memory.
// ---------------------------------------------------------------------------
bool open3d::freeDev(double** d, const string& name)
{
    cudaError_t status = cudaSuccess;

    status = cudaFree(*d);

    if (status == cudaSuccess)
    {
        *d = NULL;
        return true;
    }

    cout << "status: " << cudaGetErrorString(status) << endl;
    cout << "Failed to free device vector" << name << endl;

    return false;
}

// update the device memory on demand
bool open3d::UpdateDeviceMemory(double **d_data,
    const double* const data,
    const size_t& size) {
    cudaError_t status = cudaSuccess;

    if (*d_data != NULL) {
        status = cudaFree(*d_data);
        if (cudaSuccess != status) return false;

        *d_data = NULL;
    }

    status = cudaMalloc((void **)d_data, size * sizeof(double));
    if (cudaSuccess != status) return false;

    status = cudaMemcpy(*d_data, data, size, cudaMemcpyHostToDevice);
    if (cudaSuccess != status) return false;

    return true;
}  // namespace geometry
