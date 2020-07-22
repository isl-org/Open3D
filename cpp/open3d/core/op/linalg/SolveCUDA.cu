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

// https://
// software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/lapacke_sgesv_row.c.htm

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <stdio.h>
#include <stdlib.h>

#include "open3d/core/op/linalg/Solve.h"

namespace open3d {
namespace core {
class CuSolverContext {
public:
    static std::shared_ptr<CuSolverContext> GetInstance() {
        if (instance_ == nullptr) {
            instance_ = std::make_shared<CuSolverContext>();
        }
        return instance_;
    };

    CuSolverContext() {
        if (cusolverDnCreate(&handle_) != CUSOLVER_STATUS_SUCCESS) {
            utility::LogError("Unable to create cuSolver handle");
        }
        utility::LogInfo("Instance created");
    }
    ~CuSolverContext() {
        if (cusolverDnDestroy(handle_) != CUSOLVER_STATUS_SUCCESS) {
            utility::LogError("Unable to destroy cuSolver handle");
        }
    }

    cusolverDnHandle_t& GetHandle() { return handle_; }

private:
    cusolverDnHandle_t handle_;

    static std::shared_ptr<CuSolverContext> instance_;
};

std::shared_ptr<CuSolverContext> CuSolverContext::instance_ =
        CuSolverContext::GetInstance();

void SolveCUDA(Dtype dtype,
               void* A_data,
               void* B_data,
               void* ipiv_data,
               void* X_data,
               int n,
               int m) {
    cusolverDnHandle_t handle = CuSolverContext::GetInstance()->GetHandle();
    int niters;
    int* dinfo = static_cast<int*>(
            MemoryManager::Malloc(sizeof(int), Device("CUDA:0")));

    switch (dtype) {
        case Dtype::Float32: {
            size_t byte_size;
            if (CUSOLVER_STATUS_SUCCESS !=
                cusolverDnSSgesv_bufferSize(handle, n, m, NULL, n, NULL, NULL,
                                            n, NULL, n, NULL, &byte_size)) {
                utility::LogError("Unable to get workspace byte size");
            }

            void* workspace =
                    MemoryManager::Malloc(byte_size, Device("CUDA:0"));

            int status = cusolverDnSSgesv(
                    handle, n, m, static_cast<float*>(A_data), n,
                    static_cast<int*>(ipiv_data), static_cast<float*>(B_data),
                    n, static_cast<float*>(X_data), n, workspace, byte_size,
                    &niters, dinfo);
            if (status != CUSOLVER_STATUS_SUCCESS) {
                utility::LogError("SSgesv failed with error code = {}", status);
            }
            break;
        }

        case Dtype::Float64: {
            size_t byte_size;
            if (CUSOLVER_STATUS_SUCCESS !=
                cusolverDnDDgesv_bufferSize(handle, n, m, NULL, n, NULL, NULL,
                                            n, NULL, n, NULL, &byte_size)) {
                utility::LogError("Unable to get workspace byte size");
            }

            void* workspace =
                    MemoryManager::Malloc(byte_size, Device("CUDA:0"));

            int status = cusolverDnDDgesv(
                    handle, n, m, static_cast<double*>(A_data), n,
                    static_cast<int*>(ipiv_data), static_cast<double*>(B_data),
                    n, static_cast<double*>(X_data), n, workspace, byte_size,
                    &niters, dinfo);
            if (status != CUSOLVER_STATUS_SUCCESS) {
                utility::LogError("DDgesv failed with error code = {}", status);
            }
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
