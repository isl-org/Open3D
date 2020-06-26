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

#include "open3d/core/algebra/Solver.h"

#include <magma_v2.h>
#include <stdio.h>
#include <stdlib.h>

namespace open3d {
namespace core {
namespace _detail {

Tensor SolveCUDA(const Tensor& A, const Tensor& B) {
    // Check dimensions
    SizeVector A_shape = A.GetShape();
    SizeVector B_shape = B.GetShape();

    // A(n x n) X = B(n x m)
    if (A_shape.size() != 2) {
        utility::LogError("Tensor A must be 2D, but got {}D", A_shape.size());
    }
    if (A_shape[0] != A_shape[1]) {
        utility::LogError("Tensor A must be square, but got {} x {}",
                          A_shape[0], A_shape[1]);
    }
    if (B_shape.size() != 2) {
        utility::LogError("Tensor B must be 2D, but got {}D", B_shape.size());
    }
    if (A_shape[1] != B_shape[0]) {
        utility::LogError("Tensor A columns {} mismatch with Tensor B rows {}",
                          A_shape[1], B_shape[0]);
    }

    int n = A_shape[0], m = B_shape[1];
    int info;

    // TODO: dtype and device check
    Tensor ipiv = Tensor::Zeros({n}, Dtype::Int32, Device("CPU:0"));

    // we need copies, as the solver will override A and B in matrix
    // decomposition
    // LAPACK follows column major, so we need to transpose too
    Tensor A_copy = A.T().Copy(A.GetDevice());
    Tensor B_copy = B.T().Copy(A.GetDevice());

    void* A_data = A_copy.GetDataPtr();
    void* B_data = B_copy.GetDataPtr();
    void* ipiv_data = ipiv.GetDataPtr();

    // clang-format off
    magma_init();
    std::cout << "initialized\n";
    magma_sgesv_gpu(n, m,
                static_cast<float*>(A_data), n,
                static_cast<int*>(ipiv_data),
                static_cast<float*>(B_data), n,
                &info);
    std::cout << "sgesv finished\n";
    magma_finalize();
    std::cout << "finalized\n";
    // clang-format on
    return B_copy.T();
}
}  // namespace _detail
}  // namespace core
}  // namespace open3d