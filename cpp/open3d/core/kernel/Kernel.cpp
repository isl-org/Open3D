// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/kernel/Kernel.h"

#include <cmath>
#include <vector>

#include "open3d/core/linalg/LinalgHeadersCPU.h"
#include "open3d/utility/Logging.h"

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
