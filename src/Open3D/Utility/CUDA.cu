
#include "CUDA.cuh"

#include <iostream>
using namespace std;

// ----------------------------------------------------------------------------
// Diplay info about the specified device.
// ----------------------------------------------------------------------------
void DeviceInfo(const int& devID)
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
bool AlocateHstMemory(double** h, const int& numElements, const string& name)
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
bool AlocateDevMemory(double** d, const int& numElements, const string& name)
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
void RandInit(double* h, const int& numElements)
{
    for (int i = 0; i < numElements; ++i)
        h[i] = rand()/(double)RAND_MAX;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
bool CopyHst2DevMemory(double* h, double* d, const int& numElements)
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
// ---------------------------------------------------------------------------
bool CopyDev2HstMemory(double* d, double* h, const int& numElements)
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
// ---------------------------------------------------------------------------
bool freeDev(double** d, const string& name)
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
