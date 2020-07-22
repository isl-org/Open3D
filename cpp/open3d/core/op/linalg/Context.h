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

#pragma once

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <memory>

namespace open3d {
namespace core {
class CuSolverContext {
public:
    static std::shared_ptr<CuSolverContext> GetInstance();
    CuSolverContext();
    ~CuSolverContext();

    cusolverDnHandle_t& GetHandle() { return handle_; }

private:
    cusolverDnHandle_t handle_;

    static std::shared_ptr<CuSolverContext> instance_;
};

class CuBLASContext {
public:
    static std::shared_ptr<CuBLASContext> GetInstance();

    CuBLASContext();
    ~CuBLASContext();

    cublasHandle_t& GetHandle() { return handle_; }

private:
    cublasHandle_t handle_;

    static std::shared_ptr<CuBLASContext> instance_;
};
}  // namespace core
}  // namespace open3d
