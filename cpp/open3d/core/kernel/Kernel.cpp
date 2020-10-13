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

#include "open3d/core/kernel/Kernel.h"

#include <cmath>
#include <vector>

#include "open3d/core/linalg/LinalgHeadersCPU.h"
#include "open3d/utility/Console.h"

namespace open3d {
namespace core {
namespace kernel {

void TestLinalgIntegration() {
    // Blas
    std::vector<double> A{1.0, 2.0, 1.0, -3.0, 4.0, -1.0};
    std::vector<double> B{1.0, 2.0, 1.0, -3.0, 4.0, -1.0};
    std::vector<double> C{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, 3, 3, 2, 1, A.data(),
                3, B.data(), 3, 2, C.data(), 3);
    utility::LogInfo("TestBlas Done.");

    // Lapack
    int64_t m = 6;
    int64_t n = 5;
    int64_t lda = m;
    int64_t ldu = m;
    int64_t ldvt = n;
    std::vector<float> superb(std::min(m, n) - 1);
    std::vector<float> s(n);
    std::vector<float> u(ldu * m);
    std::vector<float> vt(ldvt * n);
    std::vector<float> a{8.79f, 6.11f,  -9.15f, 9.57f, -3.49f, 9.84f,
                         9.93f, 6.91f,  -7.93f, 1.64f, 4.02f,  0.15f,
                         9.83f, 5.04f,  4.86f,  8.83f, 9.80f,  -8.99f,
                         5.45f, -0.27f, 4.85f,  0.74f, 10.00f, -6.02f,
                         3.16f, 7.98f,  3.01f,  5.80f, 4.27f,  -5.31f};
    LAPACKE_sgesvd(LAPACK_COL_MAJOR, 'A', 'A', m, n, a.data(), lda, s.data(),
                   u.data(), ldu, vt.data(), ldvt, superb.data());
    utility::LogInfo("TestLapack Done.");
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
