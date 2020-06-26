// https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/sgesv_ex.c.htm

/*******************************************************************************
 *  Copyright (C) 2009-2015 Intel Corporation. All Rights Reserved.
 *  The information and material ("Material") provided below is owned by Intel
 *  Corporation or its suppliers or licensors, and title to such Material
 *remains with Intel Corporation or its suppliers or licensors. The Material
 *contains proprietary information of Intel or its suppliers and licensors. The
 *Material is protected by worldwide copyright laws and treaty provisions. No
 *part of the Material may be copied, reproduced, published, uploaded, posted,
 *  transmitted, or distributed in any way without Intel's prior express written
 *  permission. No license under any patent, copyright or other intellectual
 *  property rights in the Material is granted to or conferred upon you, either
 *  expressly, by implication, inducement, estoppel or otherwise. Any license
 *  under such intellectual property rights must be express and approved by
 *Intel in writing.
 *
 ********************************************************************************
 */
/*
   SGESV Example.
   ==============

   The program computes the solution to the system of linear
   equations with a square matrix A and multiple
   right-hand sides B, where A is the coefficient matrix:

     6.80  -6.05  -0.45   8.32  -9.67
    -2.11  -3.30   2.58   2.71  -5.14
     5.66   5.36  -2.70   4.35  -7.26
     5.97  -4.44   0.27  -7.17   6.08
     8.23   1.08   9.04   2.14  -6.87

   and B is the right-hand side matrix:

     4.02  -1.56   9.81
     6.19   4.00  -4.09
    -8.22  -8.67  -4.57
    -7.57   1.75  -8.61
    -3.03   2.86   8.99

   Description.
   ============

   The routine solves for X the system of linear equations A*X = B,
   where A is an n-by-n matrix, the columns of matrix B are individual
   right-hand sides, and the columns of X are the corresponding
   solutions.

   The LU decomposition with partial pivoting and row interchanges is
   used to factor A as A = P*L*U, where P is a permutation matrix, L
   is unit lower triangular, and U is upper triangular. The factored
   form of A is then used to solve the system of equations A*X = B.

   Example Program Results.
   ========================

 SGESV Example Program Results

 Solution
  -0.80  -0.39   0.96
  -0.70  -0.55   0.22
   0.59   0.84   1.90
   1.32  -0.10   5.36
   0.57   0.11   4.04

 Details of LU factorization
   8.23   1.08   9.04   2.14  -6.87
   0.83  -6.94  -7.92   6.55  -3.99
   0.69  -0.67 -14.18   7.24  -5.19
   0.73   0.75   0.02 -13.82  14.19
  -0.26   0.44  -0.59  -0.34  -3.43

 Pivot indices
      5      5      3      4      5
*/
#include <mkl.h>
#include <stdio.h>
#include <stdlib.h>

/* SGESV prototype */
extern void sgesv(int* n,
                  int* nrhs,
                  float* a,
                  int* lda,
                  int* ipiv,
                  float* b,
                  int* ldb,
                  int* info);
/* Auxiliary routines prototypes */
extern void print_matrix(char* desc, int m, int n, float* a, int lda);
extern void print_int_vector(char* desc, int n, int* a);

/* Parameters */
#define N 5
#define NRHS 3
#define LDA N
#define LDB N

/* Main program */
int main() {
    /* Locals */
    int n = N, nrhs = NRHS, lda = LDA, ldb = LDB, info;
    /* Local arrays */
    int ipiv[N];
    float a[LDA * N] = {6.80f,  -2.11f, 5.66f, 5.97f,  8.23f,  -6.05f, -3.30f,
                        5.36f,  -4.44f, 1.08f, -0.45f, 2.58f,  -2.70f, 0.27f,
                        9.04f,  8.32f,  2.71f, 4.35f,  -7.17f, 2.14f,  -9.67f,
                        -5.14f, -7.26f, 6.08f, -6.87f};
    float b[LDB * NRHS] = {4.02f,  6.19f,  -8.22f, -7.57f, -3.03f,
                           -1.56f, 4.00f,  -8.67f, 1.75f,  2.86f,
                           9.81f,  -4.09f, -4.57f, -8.61f, 8.99f};
    /* Executable statements */
    printf(" SGESV Example Program Results\n");
    /* Solve the equations A*X = B */
    sgesv(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    /* Check for the exact singularity */
    if (info > 0) {
        printf("The diagonal element of the triangular factor of A,\n");
        printf("U(%i,%i) is zero, so that A is singular;\n", info, info);
        printf("the solution could not be computed.\n");
        exit(1);
    }
    /* Print solution */
    print_matrix("Solution", n, nrhs, b, ldb);
    /* Print details of LU factorization */
    print_matrix("Details of LU factorization", n, n, a, lda);
    /* Print pivot indices */
    print_int_vector("Pivot indices", n, ipiv);
    exit(0);
} /* End of SGESV Example */

/* Auxiliary routine: printing a matrix */
void print_matrix(char* desc, int m, int n, float* a, int lda) {
    int i, j;
    printf("\n %s\n", desc);
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) printf(" %6.2f", a[i + j * lda]);
        printf("\n");
    }
}

/* Auxiliary routine: printing a vector of integers */
void print_int_vector(char* desc, int n, int* a) {
    int j;
    printf("\n %s\n", desc);
    for (j = 0; j < n; j++) printf(" %6i", a[j]);
    printf("\n");
}
