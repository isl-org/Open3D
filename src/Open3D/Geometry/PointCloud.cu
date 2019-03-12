// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include <stdio.h>
#include <iostream>
using namespace std;

#include "Open3D/Types/Vector3f.h"
using namespace open3d;

bool AlocateHstMemory(float** h, const int& numElements, const std::string& name);
bool AlocateDevMemory(float** d, const int& numElements, const std::string& name);
void RandInit(float* h, const int& numElements);
bool CopyHst2DevMemory(float* h, float* d, const int& numElements);
bool CopyDev2HstMemory(float* d, float* h, const int& numElements);
bool freeDev(float* d_A);

// ---------------------------------------------------------------------------
// dummy kernel
// ---------------------------------------------------------------------------
__global__ void dummy(float* data, int size, float* sums) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    Vector3f* points = (Vector3f*)data;

    // // nr points / thread
    // int chunkSize = size / gridDim.x / blockDim.x;

    // // index 0 of points corresponding to current thread
    // int globalOffset = gid * chunkSize;

    // Vector3f* localPoints = &points[globalOffset];

    // Vector3f point = localPoints[gid];
    Vector3f point = points[gid];

    float value = point[0] + point[1] + point[2];

    printf("value = %3.3f\n", value);
}

// ---------------------------------------------------------------------------
// helper function calls the dummy kernel
// ---------------------------------------------------------------------------
void dummyHost() {
    cout << "dummyHost::START" << endl;
    cout << endl;

    // Error code to check return values for CUDA calls
    cudaError_t status = cudaSuccess;

    // dummy<<<10, 10>>>();
    // cudaDeviceSynchronize();
    // status = cudaGetLastError();

    // if (status != cudaSuccess)
    // {
    //     cout << "status: " << cudaGetErrorString(status) << endl;
    //     cout << "Failed to launch dummy kernel" << endl;
    // }

    int numElements = 1 << 10;
    cout << "nr. of points:" << numElements << endl;

    // host memory
    float *h_A = NULL;
    float *h_C = NULL;

    // device memory
    float *d_A = NULL;
    float *d_C = NULL;

    if (!AlocateHstMemory(&h_A, numElements, "h_A")) exit(1);
    if (!AlocateHstMemory(&h_C, numElements, "h_C")) exit(1);

    RandInit(h_A, numElements);

    if (!AlocateDevMemory(&d_A, numElements, "d_A")) exit(1);
    if (!AlocateDevMemory(&d_C, numElements, "d_C")) exit(1);

    // Copy input to the device
    CopyHst2DevMemory(h_A, d_A, numElements);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;

    cout << "CUDA kernel launch with " << blocksPerGrid;
    cout << " blocks of " << threadsPerBlock << " threads" << endl;

    dummy<<<blocksPerGrid, threadsPerBlock>>>(d_A, numElements, d_C);
    cudaDeviceSynchronize();
    status = cudaGetLastError();

    if (status != cudaSuccess)
    {
        cout << "status: " << cudaGetErrorString(status) << endl;
        cout << "Failed to launch vectorAdd kernel" << endl;
        exit(1);
    }

    // Copy results to the host
    CopyDev2HstMemory(d_C, h_C, numElements);

    // Free device global memory
    freeDev(d_A);
    freeDev(d_C);

    // Free host memory
    free(h_A);
    free(h_C);

    cout << endl;
    cout << "dummyHost::END" << endl;
}

// ---------------------------------------------------------------------------
// Alocate host memory and perform validation.
// ---------------------------------------------------------------------------
bool AlocateHstMemory(float** h, const int& numElements, const string& name)
{
    size_t size = numElements * sizeof(float);

    *h = (float *)malloc(size);

    if (*h != NULL)
        return true;

    cout << "Failed to allocate host memory: " << name << endl;

    return false;
}

// ---------------------------------------------------------------------------
// Alocate device memory and perform validation.
// ---------------------------------------------------------------------------
bool AlocateDevMemory(float** d, const int& numElements, const string& name)
{
    cudaError_t status = cudaSuccess;

    size_t size = numElements * sizeof(float);

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
void RandInit(float* h, const int& numElements)
{
    for (int i = 0; i < numElements; ++i)
        h[i] = rand()/(float)RAND_MAX;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
bool CopyHst2DevMemory(float* h, float* d, const int& numElements)
{
    cudaError_t status = cudaSuccess;

    size_t size = numElements * sizeof(float);

    cout << "Copy host memory to the CUDA device." << endl;
    status = cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);

    if (status == cudaSuccess)
        return true;

    cout << "status: " << cudaGetErrorString(status) << endl;
    cout << "Failed to copy host memory to the CUDA device." << endl;

    return false;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
bool CopyDev2HstMemory(float* d, float* h, const int& numElements)
{
    cudaError_t status = cudaSuccess;

    size_t size = numElements * sizeof(float);

    cout << "Copy device memory to the host." << endl;
    status = cudaMemcpy(h, d, size, cudaMemcpyDeviceToHost);

    if (status == cudaSuccess)
        return true;

    cout << "status: " << cudaGetErrorString(status) << endl;
    cout << "Failed to copy device memory to the host." << endl;

    return false;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
bool freeDev(float* d_A)
{
    cudaError_t status = cudaSuccess;

    status = cudaFree(d_A);

    if (status == cudaSuccess)
        return true;

    cout << "status: " << cudaGetErrorString(status) << endl;
    cout << "Failed to free device vector A" << endl;

    return false;
}
