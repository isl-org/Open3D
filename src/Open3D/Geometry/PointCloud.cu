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
#include "Open3D/Types/Matrix3f.h"
using namespace open3d;

#include <iostream>
using namespace std;

// ---------------------------------------------------------------------------
// dummy kernel
// ---------------------------------------------------------------------------
__global__ void dummy(float* data, int nrPoints, float* output) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    Vector3f* points = (Vector3f*)data;
    Matrix3f* cumulants = (Matrix3f*)output;

    Vector3f p = points[gid];
    Matrix3f c = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    c[0][0] += p[0];
    c[0][1] += p[1];
    c[0][2] += p[2];
    c[1][0] += p[0] * p[0];
    c[1][1] += p[0] * p[1];
    c[1][2] += p[0] * p[2];
    c[2][0] += p[1] * p[1];
    c[2][1] += p[1] * p[2];
    c[2][2] += p[2] * p[2];

    printf("%4d: %+6.3f %+6.3f %+6.3f\n      %+6.3f %+6.3f %+6.3f\n      %+6.3f %+6.3f %+6.3f\n",
        gid, c[0][0], c[0][1], c[0][2], c[1][0], c[1][1], c[1][2], c[2][0], c[2][1], c[2][2]);

    cumulants[gid][0][0] = c[0][0];
    cumulants[gid][0][1] = c[0][1];
    cumulants[gid][0][2] = c[0][2];

    cumulants[gid][1][0] = c[1][0];
    cumulants[gid][1][1] = c[1][1];
    cumulants[gid][1][2] = c[1][2];

    cumulants[gid][2][0] = c[2][0];
    cumulants[gid][2][1] = c[2][1];
    cumulants[gid][2][2] = c[2][2];
}

// ---------------------------------------------------------------------------
// helper function calls the dummy kernel
// ---------------------------------------------------------------------------
void dummyGPU(float* const d_A, const int& nrPoints, float* const d_C) {
    cout << "dummyGPU::START" << endl;
    cout << endl;

    // Launch the dummy CUDA kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(nrPoints + threadsPerBlock - 1) / threadsPerBlock;

    cout << "CUDA kernel launch with " << blocksPerGrid;
    cout << " blocks of " << threadsPerBlock << " threads" << endl;

    dummy<<<blocksPerGrid, threadsPerBlock>>>(d_A, nrPoints, d_C);
    cudaDeviceSynchronize();

    cout << endl;
    cout << "dummyGPU::END" << endl;
}
