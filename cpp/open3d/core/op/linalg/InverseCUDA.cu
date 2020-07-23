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
#include <stdlib.h>

#include "open3d/core/op/linalg/Context.h"
#include "open3d/core/op/linalg/Inverse.h"

namespace open3d {
namespace core {

// https://stackoverflow.com/questions/50892906/what-is-the-most-efficient-way-to-compute-the-inverse-of-a-general-matrix-using
// https://stackoverflow.com/questions/28794010/solving-dense-linear-systems-ax-b-with-cuda
void InverseCUDA(
        Dtype dtype, void* A_data, void* ipiv_data, void* output_data, int n) {
    cusolverDnHandle_t handle = CuSolverContext::GetInstance()->GetHandle();
    int* dinfo = static_cast<int*>(
            MemoryManager::Malloc(sizeof(int), Device("CUDA:0")));

    switch (dtype) {
        case Dtype::Float32: {
            int len;
            if (CUSOLVER_STATUS_SUCCESS !=
                cusolverDnSgetrf_bufferSize(handle, n, n, NULL, n, &len)) {
                utility::LogError("Unable to get workspace byte size");
            }

            void* workspace = MemoryManager::Malloc(len * sizeof(float),
                                                    Device("CUDA:0"));

            int status =
                    cusolverDnSgetrf(handle, n, n, static_cast<float*>(A_data),
                                     n, static_cast<float*>(workspace),
                                     static_cast<int*>(ipiv_data), dinfo);
            if (status != CUSOLVER_STATUS_SUCCESS) {
                int dinfoh;
                MemoryManager::MemcpyToHost(&dinfoh, dinfo, Device("CUDA:0"),
                                            sizeof(int));
                utility::LogError(
                        "Sgetrf failed with error code = {}, dinfo = {}",
                        status, dinfoh);
            }

            status = cusolverDnSgetrs(
                    handle, CUBLAS_OP_N, n, n, static_cast<float*>(A_data), n,
                    static_cast<int*>(ipiv_data),
                    static_cast<float*>(output_data), n, dinfo);
            if (status != CUSOLVER_STATUS_SUCCESS) {
                int dinfoh;
                MemoryManager::MemcpyToHost(&dinfoh, dinfo, Device("CUDA:0"),
                                            sizeof(int));
                utility::LogError(
                        "Sgetrf failed with error code = {}, dinfo = {}",
                        status, dinfoh);
            }

            MemoryManager::Free(workspace, Device("CUDA:0"));
            break;
        }

        case Dtype::Float64: {
            int len;
            if (CUSOLVER_STATUS_SUCCESS !=
                cusolverDnDgetrf_bufferSize(handle, n, n, NULL, n, &len)) {
                utility::LogError("Unable to get workspace byte size");
            }

            void* workspace = MemoryManager::Malloc(len * sizeof(double),
                                                    Device("CUDA:0"));

            int status =
                    cusolverDnDgetrf(handle, n, n, static_cast<double*>(A_data),
                                     n, static_cast<double*>(workspace),
                                     static_cast<int*>(ipiv_data), dinfo);
            if (status != CUSOLVER_STATUS_SUCCESS) {
                int dinfoh;
                MemoryManager::MemcpyToHost(&dinfoh, dinfo, Device("CUDA:0"),
                                            sizeof(int));
                utility::LogError(
                        "Sgetrf failed with error code = {}, dinfo = {}",
                        status, dinfoh);
            }

            status = cusolverDnDgetrs(
                    handle, CUBLAS_OP_N, n, n, static_cast<double*>(A_data), n,
                    static_cast<int*>(ipiv_data),
                    static_cast<double*>(output_data), n, dinfo);
            if (status != CUSOLVER_STATUS_SUCCESS) {
                int dinfoh;
                MemoryManager::MemcpyToHost(&dinfoh, dinfo, Device("CUDA:0"),
                                            sizeof(int));
                utility::LogError(
                        "Sgetrf failed with error code = {}, dinfo = {}",
                        status, dinfoh);
            }

            MemoryManager::Free(workspace, Device("CUDA:0"));
            break;
        }
        default: {  // should never reach here
            utility::LogError("Unsupported dtype {} in CPU backend.",
                              DtypeUtil::ToString(dtype));
        }
    }

    MemoryManager::Free(dinfo, Device("CUDA:0"));
}

}  // namespace core
}  // namespace open3d
