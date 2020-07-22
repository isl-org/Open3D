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

#include "open3d/core/kernel/BinaryEW.h"

#ifdef _WIN32
#include <complex>
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#include <lapacke.h>
#endif

#include <cblas.h>

#include <cmath>
#include <vector>

#include "open3d/core/ShapeUtil.h"
#include "open3d/core/Tensor.h"
#include "open3d/utility/Console.h"

namespace open3d {
namespace core {
namespace kernel {

const std::unordered_set<BinaryEWOpCode, utility::hash_enum_class>
        s_boolean_binary_ew_op_codes{
                BinaryEWOpCode::LogicalAnd, BinaryEWOpCode::LogicalOr,
                BinaryEWOpCode::LogicalXor, BinaryEWOpCode::Gt,
                BinaryEWOpCode::Lt,         BinaryEWOpCode::Ge,
                BinaryEWOpCode::Le,         BinaryEWOpCode::Eq,
                BinaryEWOpCode::Ne,
        };

void DummyOpenBlasTest() {
    int i = 0;
    double A[6] = {1.0, 2.0, 1.0, -3.0, 4.0, -1.0};
    double B[6] = {1.0, 2.0, 1.0, -3.0, 4.0, -1.0};
    double C[9] = {.5, .5, .5, .5, .5, .5, .5, .5, .5};
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, 3, 3, 2, 1, A, 3, B, 3,
                2, C, 3);
    for (i = 0; i < 9; i++) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;
}

void print_matrix(char* desc, int m, int n, float* a, int lda) {
    int i, j;
    printf("\n %s\n", desc);
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) printf(" %6.2f", a[i + j * lda]);
        printf("\n");
    }
}

#ifdef _WIN32
#define M 6
#define N 5
#define LDA M
#define LDU M
#define LDVT N

void DummyLapackTest() {
    /* Locals */
    int m = M, n = N, lda = LDA, ldu = LDU, ldvt = LDVT, info;
    float superb[std::min(M, N) - 1];
    /* Local arrays */
    float s[N], u[LDU * M], vt[LDVT * N];
    float a[LDA * N] = {8.79f, 6.11f,  -9.15f, 9.57f, -3.49f, 9.84f,
                        9.93f, 6.91f,  -7.93f, 1.64f, 4.02f,  0.15f,
                        9.83f, 5.04f,  4.86f,  8.83f, 9.80f,  -8.99f,
                        5.45f, -0.27f, 4.85f,  0.74f, 10.00f, -6.02f,
                        3.16f, 7.98f,  3.01f,  5.80f, 4.27f,  -5.31f};
    /* Executable statements */
    printf("LAPACKE_sgesvd (column-major, high-level) Example Program "
           "Results\n");
    /* Compute SVD */
    info = LAPACKE_sgesvd(LAPACK_COL_MAJOR, 'A', 'A', m, n, a, lda, s, u, ldu,
                          vt, ldvt, superb);
    /* Check for convergence */
    if (info > 0) {
        printf("The algorithm computing SVD failed to converge.\n");
        exit(1);
    }
    /* Print singular values */
    print_matrix("Singular values", 1, n, s, 1);
    /* Print left singular vectors */
    print_matrix("Left singular vectors (stored columnwise)", m, n, u, ldu);
    /* Print right singular vectors */
    print_matrix("Right singular vectors (stored rowwise)", n, n, vt, ldvt);
}
#endif

void BinaryEW(const Tensor& lhs,
              const Tensor& rhs,
              Tensor& dst,
              BinaryEWOpCode op_code) {
    DummyOpenBlasTest();
#ifdef _WIN32
    DummyLapackTest();
#endif

    // lhs, rhs and dst must be on the same device.
    for (auto device :
         std::vector<Device>({rhs.GetDevice(), dst.GetDevice()})) {
        if (lhs.GetDevice() != device) {
            utility::LogError("Device mismatch {} != {}.",
                              lhs.GetDevice().ToString(), device.ToString());
        }
    }

    // broadcast(lhs.shape, rhs.shape) must be dst.shape.
    const SizeVector broadcasted_input_shape =
            shape_util::BroadcastedShape(lhs.GetShape(), rhs.GetShape());
    if (broadcasted_input_shape != dst.GetShape()) {
        utility::LogError(
                "The broadcasted input shape {} does not match the output "
                "shape {}.",
                broadcasted_input_shape, dst.GetShape());
    }

    Device::DeviceType device_type = lhs.GetDevice().GetType();
    if (device_type == Device::DeviceType::CPU) {
        BinaryEWCPU(lhs, rhs, dst, op_code);
    } else if (device_type == Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
        BinaryEWCUDA(lhs, rhs, dst, op_code);
#else
        utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
    } else {
        utility::LogError("BinaryEW: Unimplemented device");
    }
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
