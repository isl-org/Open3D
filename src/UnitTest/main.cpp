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

#include <gtest/gtest.h>

#include "TestUtility/Print.h"
#include "TestUtility/Rand.h"
#include "TestUtility/Raw.h"

#include <string>
using namespace std;

#include <cuda.h>
#include <cuda_runtime.h>

#include "Open3D/Types/Vector3f.h"
#include "Open3D/Types/Matrix3f.h"
using namespace open3d;

#include "Open3D/Utility/CUDA.cuh"

extern void dummyGPU(float* const d_A, const int& nrPoints, float* const d_C);

// ----------------------------------------------------------------------------
//
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

int main(int argc, char **argv) {
    /*/// original
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
    /*/// test
    int devID = 0;
    cudaSetDevice(devID);

    DeviceInfo(devID);

    // Error code to check return values for CUDA calls
    cudaError_t status = cudaSuccess;

    // nr. of dimensions
    int nrPoints = 1 << 8;
    cout << "nr. of points:" << nrPoints << endl;

    int inputSize = nrPoints * Vector3f::SIZE;
    int outputSize = nrPoints * Matrix3f::SIZE;

    // host memory
    float *h_A = NULL;
    float *h_C = NULL;

    // device memory
    float *d_A = NULL;
    float *d_C = NULL;

    if (!AlocateHstMemory(&h_A, inputSize, "h_A")) exit(1);
    if (!AlocateHstMemory(&h_C, outputSize, "h_C")) exit(1);

    RandInit(h_A, inputSize);

    if (!AlocateDevMemory(&d_A, inputSize, "d_A")) exit(1);
    if (!AlocateDevMemory(&d_C, outputSize, "d_C")) exit(1);

    // Copy input to the device
    CopyHst2DevMemory(h_A, d_A, inputSize);

    dummyGPU(d_A, nrPoints, d_C);
    status = cudaGetLastError();

    if (status != cudaSuccess)
    {
        cout << "status: " << cudaGetErrorString(status) << endl;
        cout << "Failed to launch vectorAdd kernel" << endl;
        exit(1);
    }

    // Copy results to the host
    CopyDev2HstMemory(d_C, h_C, outputSize);

    Matrix3f* cumulants = (Matrix3f*)h_C;
    cout << endl;
    cout << endl;
    for (int i = 0; i < nrPoints; i++)
    {
        Matrix3f c = cumulants[i];
        printf("%4d: %+6.3f %+6.3f %+6.3f\n      %+6.3f %+6.3f %+6.3f\n      %+6.3f %+6.3f %+6.3f\n",
        i, c[0][0], c[0][1], c[0][2], c[1][0], c[1][1], c[1][2], c[2][0], c[2][1], c[2][2]);
    }

    // Free device global memory
    freeDev(&d_A, "d_A");
    freeDev(&d_C, "d_C");

    // Free host memory
    free(h_A);
    free(h_C);
    //*///
}
