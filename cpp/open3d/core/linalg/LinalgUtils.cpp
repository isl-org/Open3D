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

#include "open3d/core/linalg/LinalgUtils.h"

namespace open3d {
namespace core {

std::shared_ptr<CuSolverContext> CuSolverContext::GetInstance() {
    if (instance_ == nullptr) {
        instance_ = std::make_shared<CuSolverContext>();
    }
    return instance_;
};

CuSolverContext::CuSolverContext() {
    if (cusolverDnCreate(&handle_) != CUSOLVER_STATUS_SUCCESS) {
        utility::LogError("Unable to create cuSolver handle");
    }
}
CuSolverContext::~CuSolverContext() {
    if (cusolverDnDestroy(handle_) != CUSOLVER_STATUS_SUCCESS) {
        utility::LogError("Unable to destroy cuSolver handle");
    }
}

std::shared_ptr<CuSolverContext> CuSolverContext::instance_ = nullptr;

std::shared_ptr<CuBLASContext> CuBLASContext::GetInstance() {
    if (instance_ == nullptr) {
        instance_ = std::make_shared<CuBLASContext>();
    }
    return instance_;
};

CuBLASContext::CuBLASContext() {
    if (cublasCreate(&handle_) != CUBLAS_STATUS_SUCCESS) {
        utility::LogError("Unable to create cublas handle");
    }
}
CuBLASContext::~CuBLASContext() { cublasDestroy(handle_); }

std::shared_ptr<CuBLASContext> CuBLASContext::instance_ = nullptr;

}  // namespace core
}  // namespace open3d
