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

// https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/sgesv_ex.c.htm

#include "open3d/core/algebra/Solver.h"

#include <mkl.h>
#include <stdio.h>
#include <stdlib.h>

namespace open3d {
namespace core {
namespace detail {

void SolverCPUBackend(Dtype dtype,
                      void* A_data,
                      void* B_data,
                      void* ipiv_data,
                      int n,
                      int m) {
    int info;

    switch (dtype) {
        case Dtype::Float32: {
            // clang-format off
            sgesv_(&n, &m,
                   static_cast<float*>(A_data), &n,
                   static_cast<int*>(ipiv_data),
                   static_cast<float*>(B_data), &n,
                   &info);
            // clang-format on
            break;
        }

        case Dtype::Float64: {
            // clang-format off
            dgesv_(&n, &m,
                   static_cast<double*>(A_data), &n,
                   static_cast<int*>(ipiv_data),
                   static_cast<double*>(B_data), &n,
                   &info);
            break;
            // clang-format on
        }

        default: {  // should never reach here
            utility::LogError("Unsupported dtype {} in CPU backend.",
                              DtypeUtil::ToString(dtype));
        }
    }

}  // namespace detail

// /* SGESV prototype */
// extern void sgesv(int* n,
//                   int* nrhs,
//                   float* a,
//                   int* lda,
//                   int* ipiv,
//                   float* b,
//                   int* ldb,
//                   int* info);
// /* Auxiliary routines prototypes */
// extern void print_matrix(char* desc, int m, int n, float* a, int lda);
// extern void print_int_vector(char* desc, int n, int* a);

// /* Parameters */
// #define N 5
// #define NRHS 3
// #define LDA N
// #define LDB N

// /* Main program */
// int main() {
//     /* Locals */
//     int n = N, nrhs = NRHS, lda = LDA, ldb = LDB, info;
//     /* Local arrays */
//     int ipiv[N];
//     float a[LDA * N] = {6.80f,  -2.11f, 5.66f, 5.97f,  8.23f,  -6.05f,
//     -3.30f,
//                         5.36f,  -4.44f, 1.08f, -0.45f, 2.58f,  -2.70f, 0.27f,
//                         9.04f,  8.32f,  2.71f, 4.35f,  -7.17f, 2.14f, -9.67f,
//                         -5.14f, -7.26f, 6.08f, -6.87f};
//     float b[LDB * NRHS] = {4.02f,  6.19f,  -8.22f, -7.57f, -3.03f,
//                            -1.56f, 4.00f,  -8.67f, 1.75f,  2.86f,
//                            9.81f,  -4.09f, -4.57f, -8.61f, 8.99f};
//     /* Executable statements */
//     printf(" SGESV Example Program Results\n");
//     /* Solve the equations A*X = B */
//     sgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
//     /* Check for the exact singularity */
//     if (info > 0) {
//         printf("The diagonal element of the triangular factor of A,\n");
//         printf("U(%i,%i) is zero, so that A is singular;\n", info, info);
//         printf("the solution could not be computed.\n");
//         exit(1);
//     }
//     /* Print solution */
//     print_matrix("Solution", n, nrhs, b, ldb);
//     /* Print details of LU factorization */
//     print_matrix("Details of LU factorization", n, n, a, lda);
//     /* Print pivot indices */
//     print_int_vector("Pivot indices", n, ipiv);
//     exit(0);
// } /* End of SGESV Example */

// /* Auxiliary routine: printing a matrix */
// void print_matrix(char* desc, int m, int n, float* a, int lda) {
//     int i, j;
//     printf("\n %s\n", desc);
//     for (i = 0; i < m; i++) {
//         for (j = 0; j < n; j++) printf(" %6.2f", a[i + j * lda]);
//         printf("\n");
//     }
// }

// /* Auxiliary routine: printing a vector of integers */
// void print_int_vector(char* desc, int n, int* a) {
//     int j;
//     printf("\n %s\n", desc);
//     for (j = 0; j < n; j++) printf(" %6i", a[j]);
//     printf("\n");
// }
}  // namespace detail
}  // namespace core
}  // namespace open3d
