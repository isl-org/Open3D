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

#include "Open3D/Types/Vector3f.h"
using namespace open3d;

#include "Open3D/Utility/CUDA.cuh"

#include <iostream>
using namespace std;

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

    printf("value[%4d] = %3.3f\n", gid, value);
}

// ---------------------------------------------------------------------------
// helper function calls the dummy kernel
// ---------------------------------------------------------------------------
void dummyHost() {
    cout << "dummyHost::START" << endl;
    cout << endl;

    // Error code to check return values for CUDA calls
    cudaError_t status = cudaSuccess;

    // nr. of dimensions
    const int DIM = 3;
    int nrPoints = 1 << 10;
    cout << "nr. of points:" << nrPoints << endl;

    int size = nrPoints * DIM;

    // host memory
    float *h_A = NULL;
    float *h_C = NULL;

    // device memory
    float *d_A = NULL;
    float *d_C = NULL;

    if (!AlocateHstMemory(&h_A, size, "h_A")) exit(1);
    if (!AlocateHstMemory(&h_C, size, "h_C")) exit(1);

    RandInit(h_A, size);

    if (!AlocateDevMemory(&d_A, size, "d_A")) exit(1);
    if (!AlocateDevMemory(&d_C, size, "d_C")) exit(1);

    // Copy input to the device
    CopyHst2DevMemory(h_A, d_A, size);

    // Launch the dummy CUDA kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(nrPoints + threadsPerBlock - 1) / threadsPerBlock;

    cout << "CUDA kernel launch with " << blocksPerGrid;
    cout << " blocks of " << threadsPerBlock << " threads" << endl;

    dummy<<<blocksPerGrid, threadsPerBlock>>>(d_A, nrPoints, d_C);
    cudaDeviceSynchronize();
    status = cudaGetLastError();

    if (status != cudaSuccess)
    {
        cout << "status: " << cudaGetErrorString(status) << endl;
        cout << "Failed to launch vectorAdd kernel" << endl;
        exit(1);
    }

    // Copy results to the host
    CopyDev2HstMemory(d_C, h_C, size);

    // Free device global memory
    freeDev(&d_A, "d_A");
    freeDev(&d_C, "d_C");

    // Free host memory
    free(h_A);
    free(h_C);

    cout << endl;
    cout << "dummyHost::END" << endl;
}
