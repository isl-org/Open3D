// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/core/linalg/LinalgHeadersCPU.h"
#include "open3d/core/linalg/LinalgHeadersCUDA.h"
#include "open3d/utility/Logging.h"

#if defined(OPEN3D_DISABLE_LAPACKE)
// Minimal LAPACK/LAPACKE stubs for platforms where LAPACKE is unavailable
// (e.g. Windows ARM64). The generic templates below will LogError and return
// -1.
#ifndef LAPACK_COL_MAJOR
#define LAPACK_COL_MAJOR 102
#endif
#include <Eigen/Dense>
#endif

namespace open3d {
namespace core {
template <typename scalar_t>
inline OPEN3D_CPU_LINALG_INT getrf_cpu(int layout,
                                       OPEN3D_CPU_LINALG_INT m,
                                       OPEN3D_CPU_LINALG_INT n,
                                       scalar_t* A_data,
                                       OPEN3D_CPU_LINALG_INT lda,
                                       OPEN3D_CPU_LINALG_INT* ipiv_data) {
    utility::LogError("Unsupported data type.");
    return -1;
}

template <typename scalar_t>
inline OPEN3D_CPU_LINALG_INT getri_cpu(int layout,
                                       OPEN3D_CPU_LINALG_INT n,
                                       scalar_t* A_data,
                                       OPEN3D_CPU_LINALG_INT lda,
                                       OPEN3D_CPU_LINALG_INT* ipiv_data) {
    utility::LogError("Unsupported data type.");
    return -1;
}

template <typename scalar_t>
inline OPEN3D_CPU_LINALG_INT gesv_cpu(int layout,
                                      OPEN3D_CPU_LINALG_INT n,
                                      OPEN3D_CPU_LINALG_INT m,
                                      scalar_t* A_data,
                                      OPEN3D_CPU_LINALG_INT lda,
                                      OPEN3D_CPU_LINALG_INT* ipiv_data,
                                      scalar_t* B_data,
                                      OPEN3D_CPU_LINALG_INT ldb) {
    utility::LogError("Unsupported data type.");
    return -1;
}

template <typename scalar_t>
inline OPEN3D_CPU_LINALG_INT gels_cpu(int matrix_layout,
                                      char trans,
                                      OPEN3D_CPU_LINALG_INT m,
                                      OPEN3D_CPU_LINALG_INT n,
                                      OPEN3D_CPU_LINALG_INT nrhs,
                                      scalar_t* A_data,
                                      OPEN3D_CPU_LINALG_INT lda,
                                      scalar_t* B_data,
                                      OPEN3D_CPU_LINALG_INT ldb) {
    utility::LogError("Unsupported data type.");
    return -1;
}

template <typename scalar_t>
inline OPEN3D_CPU_LINALG_INT gesvd_cpu(int matrix_layout,
                                       char jobu,
                                       char jobvt,
                                       OPEN3D_CPU_LINALG_INT m,
                                       OPEN3D_CPU_LINALG_INT n,
                                       scalar_t* A_data,
                                       OPEN3D_CPU_LINALG_INT lda,
                                       scalar_t* S_data,
                                       scalar_t* U_data,
                                       OPEN3D_CPU_LINALG_INT ldu,
                                       scalar_t* VT_data,
                                       OPEN3D_CPU_LINALG_INT ldvt,
                                       scalar_t* superb) {
    utility::LogError("Unsupported data type.");
    return -1;
}

#if defined(OPEN3D_DISABLE_LAPACKE)
// Eigen-backed CPU LAPACK fallback for platforms without LAPACKE (Windows
// ARM64). All buffers are column-major (callers transpose in/out), which
// matches Eigen's default storage order, so data maps directly. Only float and
// double are specialized; other types fall through to the LogError generics
// above. Return value mirrors LAPACKE's `info` (0 = success, >0 = singular,
// <0 = bad argument).
namespace lapack_fallback {

template <typename scalar_t>
using ColMatrix = Eigen::
        Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
template <typename scalar_t>
using MatrixMap = Eigen::Map<ColMatrix<scalar_t>>;

// getrf: in-place LU with partial pivoting, reproducing LAPACK's packed output
// (unit-lower L below the diagonal, U on and above) and 1-based sequential row
// pivots in `ipiv` (row i was swapped with row ipiv[i], applied in order).
template <typename scalar_t>
inline OPEN3D_CPU_LINALG_INT getrf_impl(OPEN3D_CPU_LINALG_INT m,
                                        OPEN3D_CPU_LINALG_INT n,
                                        scalar_t* A,
                                        OPEN3D_CPU_LINALG_INT lda,
                                        OPEN3D_CPU_LINALG_INT* ipiv) {
    auto at = [&](OPEN3D_CPU_LINALG_INT i,
                  OPEN3D_CPU_LINALG_INT j) -> scalar_t& {
        return A[i + j * lda];  // column-major
    };
    const OPEN3D_CPU_LINALG_INT mn = std::min(m, n);
    OPEN3D_CPU_LINALG_INT info = 0;
    for (OPEN3D_CPU_LINALG_INT k = 0; k < mn; ++k) {
        // Find pivot row (largest magnitude) in column k, rows k..m-1.
        OPEN3D_CPU_LINALG_INT p = k;
        scalar_t maxval = std::abs(at(k, k));
        for (OPEN3D_CPU_LINALG_INT i = k + 1; i < m; ++i) {
            scalar_t v = std::abs(at(i, k));
            if (v > maxval) {
                maxval = v;
                p = i;
            }
        }
        ipiv[k] = p + 1;  // 1-based
        if (at(p, k) == scalar_t(0)) {
            if (info == 0) info = k + 1;  // first zero pivot (singular)
            continue;
        }
        if (p != k) {
            for (OPEN3D_CPU_LINALG_INT j = 0; j < n; ++j) {
                std::swap(at(k, j), at(p, j));
            }
        }
        const scalar_t inv_pivot = scalar_t(1) / at(k, k);
        for (OPEN3D_CPU_LINALG_INT i = k + 1; i < m; ++i) {
            at(i, k) *= inv_pivot;
        }
        for (OPEN3D_CPU_LINALG_INT j = k + 1; j < n; ++j) {
            const scalar_t akj = at(k, j);
            for (OPEN3D_CPU_LINALG_INT i = k + 1; i < m; ++i) {
                at(i, j) -= at(i, k) * akj;
            }
        }
    }
    return info;
}

// Reconstruct the original matrix from a getrf result and invert it. getri in
// LAPACK inverts the in-place factored matrix; InverseCPU calls getrf then
// getri on the same buffer, so we undo the factorization (P·L·U) and invert.
template <typename scalar_t>
inline OPEN3D_CPU_LINALG_INT getri_impl(OPEN3D_CPU_LINALG_INT n,
                                        scalar_t* A,
                                        OPEN3D_CPU_LINALG_INT lda,
                                        const OPEN3D_CPU_LINALG_INT* ipiv) {
    ColMatrix<scalar_t> L = ColMatrix<scalar_t>::Identity(n, n);
    ColMatrix<scalar_t> U = ColMatrix<scalar_t>::Zero(n, n);
    auto at = [&](OPEN3D_CPU_LINALG_INT i,
                  OPEN3D_CPU_LINALG_INT j) -> scalar_t {
        return A[i + j * lda];
    };
    for (OPEN3D_CPU_LINALG_INT j = 0; j < n; ++j) {
        for (OPEN3D_CPU_LINALG_INT i = 0; i < n; ++i) {
            if (i > j) {
                L(i, j) = at(i, j);
            } else {
                U(i, j) = at(i, j);
            }
        }
    }
    ColMatrix<scalar_t> LU = L * U;
    // Apply row interchanges (in reverse) to build P·L·U = original A.
    for (OPEN3D_CPU_LINALG_INT k = n - 1; k >= 0; --k) {
        OPEN3D_CPU_LINALG_INT p = ipiv[k] - 1;
        if (p != k) {
            LU.row(k).swap(LU.row(p));
        }
    }
    Eigen::FullPivLU<ColMatrix<scalar_t>> lu(LU);
    if (!lu.isInvertible()) {
        return 1;  // singular
    }
    ColMatrix<scalar_t> inv = lu.inverse();
    MatrixMap<scalar_t>(A, n, n) = inv;
    return 0;
}

}  // namespace lapack_fallback

template <>
inline OPEN3D_CPU_LINALG_INT getrf_cpu<float>(
        int layout,
        OPEN3D_CPU_LINALG_INT m,
        OPEN3D_CPU_LINALG_INT n,
        float* A_data,
        OPEN3D_CPU_LINALG_INT lda,
        OPEN3D_CPU_LINALG_INT* ipiv_data) {
    return lapack_fallback::getrf_impl<float>(m, n, A_data, lda, ipiv_data);
}

template <>
inline OPEN3D_CPU_LINALG_INT getrf_cpu<double>(
        int layout,
        OPEN3D_CPU_LINALG_INT m,
        OPEN3D_CPU_LINALG_INT n,
        double* A_data,
        OPEN3D_CPU_LINALG_INT lda,
        OPEN3D_CPU_LINALG_INT* ipiv_data) {
    return lapack_fallback::getrf_impl<double>(m, n, A_data, lda, ipiv_data);
}

template <>
inline OPEN3D_CPU_LINALG_INT getri_cpu<float>(
        int layout,
        OPEN3D_CPU_LINALG_INT n,
        float* A_data,
        OPEN3D_CPU_LINALG_INT lda,
        OPEN3D_CPU_LINALG_INT* ipiv_data) {
    return lapack_fallback::getri_impl<float>(n, A_data, lda, ipiv_data);
}

template <>
inline OPEN3D_CPU_LINALG_INT getri_cpu<double>(
        int layout,
        OPEN3D_CPU_LINALG_INT n,
        double* A_data,
        OPEN3D_CPU_LINALG_INT lda,
        OPEN3D_CPU_LINALG_INT* ipiv_data) {
    return lapack_fallback::getri_impl<double>(n, A_data, lda, ipiv_data);
}

// gesv: solve A·X = B. A is n×n, B is n×nrhs (column-major). Overwrites A with
// its LU factors and B with the solution X (matching LAPACKE_?gesv semantics
// closely enough for Open3D, which only consumes X).
template <>
inline OPEN3D_CPU_LINALG_INT gesv_cpu<float>(int layout,
                                             OPEN3D_CPU_LINALG_INT n,
                                             OPEN3D_CPU_LINALG_INT m,
                                             float* A_data,
                                             OPEN3D_CPU_LINALG_INT lda,
                                             OPEN3D_CPU_LINALG_INT* ipiv_data,
                                             float* B_data,
                                             OPEN3D_CPU_LINALG_INT ldb) {
    (void)ipiv_data;
    lapack_fallback::MatrixMap<float> A(A_data, n, n);
    lapack_fallback::MatrixMap<float> B(B_data, n, m);
    Eigen::FullPivLU<lapack_fallback::ColMatrix<float>> lu(A);
    if (!lu.isInvertible()) {
        return 1;  // singular; OPEN3D_LAPACK_CHECK turns info>0 into a throw
    }
    B = lu.solve(B).eval();
    return 0;
}

template <>
inline OPEN3D_CPU_LINALG_INT gesv_cpu<double>(int layout,
                                              OPEN3D_CPU_LINALG_INT n,
                                              OPEN3D_CPU_LINALG_INT m,
                                              double* A_data,
                                              OPEN3D_CPU_LINALG_INT lda,
                                              OPEN3D_CPU_LINALG_INT* ipiv_data,
                                              double* B_data,
                                              OPEN3D_CPU_LINALG_INT ldb) {
    (void)ipiv_data;
    lapack_fallback::MatrixMap<double> A(A_data, n, n);
    lapack_fallback::MatrixMap<double> B(B_data, n, m);
    Eigen::FullPivLU<lapack_fallback::ColMatrix<double>> lu(A);
    if (!lu.isInvertible()) {
        return 1;  // singular; OPEN3D_LAPACK_CHECK turns info>0 into a throw
    }
    B = lu.solve(B).eval();
    return 0;
}

// gels: least-squares solve of an over-determined A·X ≈ B via Householder QR.
// A is m×n (m >= n), B is m×nrhs. On exit the leading n rows of B hold X.
template <>
inline OPEN3D_CPU_LINALG_INT gels_cpu<float>(int layout,
                                             char trans,
                                             OPEN3D_CPU_LINALG_INT m,
                                             OPEN3D_CPU_LINALG_INT n,
                                             OPEN3D_CPU_LINALG_INT nrhs,
                                             float* A_data,
                                             OPEN3D_CPU_LINALG_INT lda,
                                             float* B_data,
                                             OPEN3D_CPU_LINALG_INT ldb) {
    (void)trans;
    lapack_fallback::MatrixMap<float> A(A_data, m, n);
    lapack_fallback::MatrixMap<float> B(B_data, ldb, nrhs);
    lapack_fallback::ColMatrix<float> X =
            A.householderQr().solve(B.topRows(m).eval());
    B.topRows(n) = X;
    return 0;
}

template <>
inline OPEN3D_CPU_LINALG_INT gels_cpu<double>(int layout,
                                              char trans,
                                              OPEN3D_CPU_LINALG_INT m,
                                              OPEN3D_CPU_LINALG_INT n,
                                              OPEN3D_CPU_LINALG_INT nrhs,
                                              double* A_data,
                                              OPEN3D_CPU_LINALG_INT lda,
                                              double* B_data,
                                              OPEN3D_CPU_LINALG_INT ldb) {
    (void)trans;
    lapack_fallback::MatrixMap<double> A(A_data, m, n);
    lapack_fallback::MatrixMap<double> B(B_data, ldb, nrhs);
    lapack_fallback::ColMatrix<double> X =
            A.householderQr().solve(B.topRows(m).eval());
    B.topRows(n) = X;
    return 0;
}

// gesvd: full SVD (jobu='A', jobvt='A'). A is m×n; writes U (m×m), singular
// values S (min(m,n)), and VT (n×n), all column-major. `superb` is unused.
template <>
inline OPEN3D_CPU_LINALG_INT gesvd_cpu<float>(int layout,
                                              char jobu,
                                              char jobvt,
                                              OPEN3D_CPU_LINALG_INT m,
                                              OPEN3D_CPU_LINALG_INT n,
                                              float* A_data,
                                              OPEN3D_CPU_LINALG_INT lda,
                                              float* S_data,
                                              float* U_data,
                                              OPEN3D_CPU_LINALG_INT ldu,
                                              float* VT_data,
                                              OPEN3D_CPU_LINALG_INT ldvt,
                                              float* superb) {
    (void)jobu;
    (void)jobvt;
    (void)superb;
    lapack_fallback::MatrixMap<float> A(A_data, m, n);
    Eigen::JacobiSVD<lapack_fallback::ColMatrix<float>> svd(
            A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    lapack_fallback::MatrixMap<float>(U_data, m, m) = svd.matrixU();
    lapack_fallback::MatrixMap<float>(VT_data, n, n) =
            svd.matrixV().transpose();
    const OPEN3D_CPU_LINALG_INT mn = std::min(m, n);
    for (OPEN3D_CPU_LINALG_INT i = 0; i < mn; ++i) {
        S_data[i] = svd.singularValues()(i);
    }
    return 0;
}

template <>
inline OPEN3D_CPU_LINALG_INT gesvd_cpu<double>(int layout,
                                               char jobu,
                                               char jobvt,
                                               OPEN3D_CPU_LINALG_INT m,
                                               OPEN3D_CPU_LINALG_INT n,
                                               double* A_data,
                                               OPEN3D_CPU_LINALG_INT lda,
                                               double* S_data,
                                               double* U_data,
                                               OPEN3D_CPU_LINALG_INT ldu,
                                               double* VT_data,
                                               OPEN3D_CPU_LINALG_INT ldvt,
                                               double* superb) {
    (void)jobu;
    (void)jobvt;
    (void)superb;
    lapack_fallback::MatrixMap<double> A(A_data, m, n);
    Eigen::JacobiSVD<lapack_fallback::ColMatrix<double>> svd(
            A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    lapack_fallback::MatrixMap<double>(U_data, m, m) = svd.matrixU();
    lapack_fallback::MatrixMap<double>(VT_data, n, n) =
            svd.matrixV().transpose();
    const OPEN3D_CPU_LINALG_INT mn = std::min(m, n);
    for (OPEN3D_CPU_LINALG_INT i = 0; i < mn; ++i) {
        S_data[i] = svd.singularValues()(i);
    }
    return 0;
}
#endif  // OPEN3D_DISABLE_LAPACKE

#if !defined(OPEN3D_DISABLE_LAPACKE)
template <>
inline OPEN3D_CPU_LINALG_INT getrf_cpu<float>(
        int layout,
        OPEN3D_CPU_LINALG_INT m,
        OPEN3D_CPU_LINALG_INT n,
        float* A_data,
        OPEN3D_CPU_LINALG_INT lda,
        OPEN3D_CPU_LINALG_INT* ipiv_data) {
    return LAPACKE_sgetrf(layout, m, n, A_data, lda, ipiv_data);
}

template <>
inline OPEN3D_CPU_LINALG_INT getrf_cpu<double>(
        int layout,
        OPEN3D_CPU_LINALG_INT m,
        OPEN3D_CPU_LINALG_INT n,
        double* A_data,
        OPEN3D_CPU_LINALG_INT lda,
        OPEN3D_CPU_LINALG_INT* ipiv_data) {
    return LAPACKE_dgetrf(layout, m, n, A_data, lda, ipiv_data);
}

template <>
inline OPEN3D_CPU_LINALG_INT getri_cpu<float>(
        int layout,
        OPEN3D_CPU_LINALG_INT n,
        float* A_data,
        OPEN3D_CPU_LINALG_INT lda,
        OPEN3D_CPU_LINALG_INT* ipiv_data) {
    return LAPACKE_sgetri(layout, n, A_data, lda, ipiv_data);
}

template <>
inline OPEN3D_CPU_LINALG_INT getri_cpu<double>(
        int layout,
        OPEN3D_CPU_LINALG_INT n,
        double* A_data,
        OPEN3D_CPU_LINALG_INT lda,
        OPEN3D_CPU_LINALG_INT* ipiv_data) {
    return LAPACKE_dgetri(layout, n, A_data, lda, ipiv_data);
}

template <>
inline OPEN3D_CPU_LINALG_INT gesv_cpu<float>(int layout,
                                             OPEN3D_CPU_LINALG_INT n,
                                             OPEN3D_CPU_LINALG_INT m,
                                             float* A_data,
                                             OPEN3D_CPU_LINALG_INT lda,
                                             OPEN3D_CPU_LINALG_INT* ipiv_data,
                                             float* B_data,
                                             OPEN3D_CPU_LINALG_INT ldb) {
    return LAPACKE_sgesv(layout, n, m, A_data, lda, ipiv_data, B_data, ldb);
}

template <>
inline OPEN3D_CPU_LINALG_INT gesv_cpu<double>(int layout,
                                              OPEN3D_CPU_LINALG_INT n,
                                              OPEN3D_CPU_LINALG_INT m,
                                              double* A_data,
                                              OPEN3D_CPU_LINALG_INT lda,
                                              OPEN3D_CPU_LINALG_INT* ipiv_data,
                                              double* B_data,
                                              OPEN3D_CPU_LINALG_INT ldb) {
    return LAPACKE_dgesv(layout, n, m, A_data, lda, ipiv_data, B_data, ldb);
}

template <>
inline OPEN3D_CPU_LINALG_INT gels_cpu<float>(int layout,
                                             char trans,
                                             OPEN3D_CPU_LINALG_INT m,
                                             OPEN3D_CPU_LINALG_INT n,
                                             OPEN3D_CPU_LINALG_INT nrhs,
                                             float* A_data,
                                             OPEN3D_CPU_LINALG_INT lda,
                                             float* B_data,
                                             OPEN3D_CPU_LINALG_INT ldb) {
    return LAPACKE_sgels(layout, trans, m, n, nrhs, A_data, lda, B_data, ldb);
}

template <>
inline OPEN3D_CPU_LINALG_INT gels_cpu<double>(int layout,
                                              char trans,
                                              OPEN3D_CPU_LINALG_INT m,
                                              OPEN3D_CPU_LINALG_INT n,
                                              OPEN3D_CPU_LINALG_INT nrhs,
                                              double* A_data,
                                              OPEN3D_CPU_LINALG_INT lda,
                                              double* B_data,
                                              OPEN3D_CPU_LINALG_INT ldb) {
    return LAPACKE_dgels(layout, trans, m, n, nrhs, A_data, lda, B_data, ldb);
}

template <>
inline OPEN3D_CPU_LINALG_INT gesvd_cpu<float>(int layout,
                                              char jobu,
                                              char jobvt,
                                              OPEN3D_CPU_LINALG_INT m,
                                              OPEN3D_CPU_LINALG_INT n,
                                              float* A_data,
                                              OPEN3D_CPU_LINALG_INT lda,
                                              float* S_data,
                                              float* U_data,
                                              OPEN3D_CPU_LINALG_INT ldu,
                                              float* VT_data,
                                              OPEN3D_CPU_LINALG_INT ldvt,
                                              float* superb) {
    return LAPACKE_sgesvd(layout, jobu, jobvt, m, n, A_data, lda, S_data,
                          U_data, ldu, VT_data, ldvt, superb);
}

template <>
inline OPEN3D_CPU_LINALG_INT gesvd_cpu<double>(int layout,
                                               char jobu,
                                               char jobvt,
                                               OPEN3D_CPU_LINALG_INT m,
                                               OPEN3D_CPU_LINALG_INT n,
                                               double* A_data,
                                               OPEN3D_CPU_LINALG_INT lda,
                                               double* S_data,
                                               double* U_data,
                                               OPEN3D_CPU_LINALG_INT ldu,
                                               double* VT_data,
                                               OPEN3D_CPU_LINALG_INT ldvt,
                                               double* superb) {
    return LAPACKE_dgesvd(layout, jobu, jobvt, m, n, A_data, lda, S_data,
                          U_data, ldu, VT_data, ldvt, superb);
}
#endif  // !OPEN3D_DISABLE_LAPACKE

#ifdef BUILD_CUDA_MODULE
template <typename scalar_t>
inline cusolverStatus_t getrf_cuda_buffersize(
        cusolverDnHandle_t handle, int m, int n, int lda, int* len) {
    utility::LogError("Unsupported data type.");
    return CUSOLVER_STATUS_INTERNAL_ERROR;
}

template <typename scalar_t>
inline cusolverStatus_t getrf_cuda(cusolverDnHandle_t handle,
                                   int m,
                                   int n,
                                   scalar_t* A_data,
                                   int lda,
                                   scalar_t* workspace,
                                   int* ipiv_data,
                                   int* dinfo) {
    utility::LogError("Unsupported data type.");
    return CUSOLVER_STATUS_INTERNAL_ERROR;
}

template <typename scalar_t>
inline cusolverStatus_t getrs_cuda(cusolverDnHandle_t handle,
                                   cublasOperation_t trans,
                                   int n,
                                   int nrhs,
                                   const scalar_t* A_data,
                                   int lda,
                                   const int* ipiv_data,
                                   scalar_t* B_data,
                                   int ldb,
                                   int* dinfo) {
    utility::LogError("Unsupported data type.");
    return CUSOLVER_STATUS_INTERNAL_ERROR;
}

template <typename scalar_t>
inline cusolverStatus_t geqrf_cuda_buffersize(
        cusolverDnHandle_t handle, int m, int n, int lda, int* len) {
    utility::LogError("Unsupported data type.");
    return CUSOLVER_STATUS_INTERNAL_ERROR;
}

template <typename scalar_t>
inline cusolverStatus_t geqrf_cuda(cusolverDnHandle_t handle,
                                   int m,
                                   int n,
                                   scalar_t* A,
                                   int lda,
                                   scalar_t* tau,
                                   scalar_t* workspace,
                                   int len,
                                   int* dinfo) {
    utility::LogError("Unsupported data type.");
    return CUSOLVER_STATUS_INTERNAL_ERROR;
}

template <typename scalar_t>
inline cusolverStatus_t ormqr_cuda_buffersize(cusolverDnHandle_t handle,
                                              cublasSideMode_t side,
                                              cublasOperation_t trans,
                                              int m,
                                              int n,
                                              int k,
                                              int lda,
                                              int ldc,
                                              int* len) {
    utility::LogError("Unsupported data type.");
    return CUSOLVER_STATUS_INTERNAL_ERROR;
}

template <typename scalar_t>
inline cusolverStatus_t ormqr_cuda(cusolverDnHandle_t handle,
                                   cublasSideMode_t side,
                                   cublasOperation_t trans,
                                   int m,
                                   int n,
                                   int k,
                                   const scalar_t* A,
                                   int lda,
                                   const scalar_t* tau,
                                   scalar_t* C,
                                   int ldc,
                                   scalar_t* workspace,
                                   int len,
                                   int* dinfo) {
    utility::LogError("Unsupported data type.");
    return CUSOLVER_STATUS_INTERNAL_ERROR;
}

template <typename scalar_t>
inline cusolverStatus_t gesvd_cuda_buffersize(cusolverDnHandle_t handle,
                                              int m,
                                              int n,
                                              int* len) {
    utility::LogError("Unsupported data type.");
    return CUSOLVER_STATUS_INTERNAL_ERROR;
}

template <typename scalar_t>
inline cusolverStatus_t gesvd_cuda(cusolverDnHandle_t handle,
                                   char jobu,
                                   char jobvt,
                                   int m,
                                   int n,
                                   scalar_t* A,
                                   int lda,
                                   scalar_t* S,
                                   scalar_t* U,
                                   int ldu,
                                   scalar_t* VT,
                                   int ldvt,
                                   scalar_t* workspace,
                                   int len,
                                   scalar_t* rwork,
                                   int* dinfo) {
    utility::LogError("Unsupported data type.");
    return CUSOLVER_STATUS_INTERNAL_ERROR;
}

template <>
inline cusolverStatus_t getrf_cuda_buffersize<float>(
        cusolverDnHandle_t handle, int m, int n, int lda, int* len) {
    return cusolverDnSgetrf_bufferSize(handle, m, n, NULL, lda, len);
}

template <>
inline cusolverStatus_t getrf_cuda_buffersize<double>(
        cusolverDnHandle_t handle, int m, int n, int lda, int* len) {
    return cusolverDnDgetrf_bufferSize(handle, m, n, NULL, lda, len);
}

template <>
inline cusolverStatus_t getrf_cuda<float>(cusolverDnHandle_t handle,
                                          int m,
                                          int n,
                                          float* A_data,
                                          int lda,
                                          float* workspace,
                                          int* ipiv_data,
                                          int* dinfo) {
    return cusolverDnSgetrf(handle, m, n, A_data, lda, workspace, ipiv_data,
                            dinfo);
}

template <>
inline cusolverStatus_t getrf_cuda<double>(cusolverDnHandle_t handle,
                                           int m,
                                           int n,
                                           double* A_data,
                                           int lda,
                                           double* workspace,
                                           int* ipiv_data,
                                           int* dinfo) {
    return cusolverDnDgetrf(handle, m, n, A_data, lda, workspace, ipiv_data,
                            dinfo);
}

template <>
inline cusolverStatus_t getrs_cuda<float>(cusolverDnHandle_t handle,
                                          cublasOperation_t trans,
                                          int n,
                                          int nrhs,
                                          const float* A_data,
                                          int lda,
                                          const int* ipiv_data,
                                          float* B_data,
                                          int ldb,
                                          int* dinfo) {
    return cusolverDnSgetrs(handle, trans, n, nrhs, A_data, lda, ipiv_data,
                            B_data, ldb, dinfo);
}

template <>
inline cusolverStatus_t getrs_cuda<double>(cusolverDnHandle_t handle,
                                           cublasOperation_t trans,
                                           int n,
                                           int nrhs,
                                           const double* A_data,
                                           int lda,
                                           const int* ipiv_data,
                                           double* B_data,
                                           int ldb,
                                           int* dinfo) {
    return cusolverDnDgetrs(handle, trans, n, nrhs, A_data, lda, ipiv_data,
                            B_data, ldb, dinfo);
}

template <>
inline cusolverStatus_t geqrf_cuda_buffersize<float>(
        cusolverDnHandle_t handle, int m, int n, int lda, int* len) {
    return cusolverDnSgeqrf_bufferSize(handle, m, n, NULL, lda, len);
}

template <>
inline cusolverStatus_t geqrf_cuda_buffersize<double>(
        cusolverDnHandle_t handle, int m, int n, int lda, int* len) {
    return cusolverDnDgeqrf_bufferSize(handle, m, n, NULL, lda, len);
}

template <>
inline cusolverStatus_t geqrf_cuda<float>(cusolverDnHandle_t handle,
                                          int m,
                                          int n,
                                          float* A,
                                          int lda,
                                          float* tau,
                                          float* workspace,
                                          int len,
                                          int* dinfo) {
    return cusolverDnSgeqrf(handle, m, n, A, lda, tau, workspace, len, dinfo);
}

template <>
inline cusolverStatus_t geqrf_cuda<double>(cusolverDnHandle_t handle,
                                           int m,
                                           int n,
                                           double* A,
                                           int lda,
                                           double* tau,
                                           double* workspace,
                                           int len,
                                           int* dinfo) {
    return cusolverDnDgeqrf(handle, m, n, A, lda, tau, workspace, len, dinfo);
}

template <>
inline cusolverStatus_t ormqr_cuda_buffersize<float>(cusolverDnHandle_t handle,
                                                     cublasSideMode_t side,
                                                     cublasOperation_t trans,
                                                     int m,
                                                     int n,
                                                     int k,
                                                     int lda,
                                                     int ldc,
                                                     int* len) {
    return cusolverDnSormqr_bufferSize(handle, side, trans, m, n, k, NULL, lda,
                                       NULL, NULL, ldc, len);
}

template <>
inline cusolverStatus_t ormqr_cuda_buffersize<double>(cusolverDnHandle_t handle,
                                                      cublasSideMode_t side,
                                                      cublasOperation_t trans,
                                                      int m,
                                                      int n,
                                                      int k,
                                                      int lda,
                                                      int ldc,
                                                      int* len) {
    return cusolverDnDormqr_bufferSize(handle, side, trans, m, n, k, NULL, lda,
                                       NULL, NULL, ldc, len);
}

template <>
inline cusolverStatus_t ormqr_cuda<float>(cusolverDnHandle_t handle,
                                          cublasSideMode_t side,
                                          cublasOperation_t trans,
                                          int m,
                                          int n,
                                          int k,
                                          const float* A,
                                          int lda,
                                          const float* tau,
                                          float* C,
                                          int ldc,
                                          float* workspace,
                                          int len,
                                          int* dinfo) {
    return cusolverDnSormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc,
                            workspace, len, dinfo);
}

template <>
inline cusolverStatus_t ormqr_cuda<double>(cusolverDnHandle_t handle,
                                           cublasSideMode_t side,
                                           cublasOperation_t trans,
                                           int m,
                                           int n,
                                           int k,
                                           const double* A,
                                           int lda,
                                           const double* tau,
                                           double* C,
                                           int ldc,
                                           double* workspace,
                                           int len,
                                           int* dinfo) {
    return cusolverDnDormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc,
                            workspace, len, dinfo);
}

template <>
inline cusolverStatus_t gesvd_cuda_buffersize<float>(cusolverDnHandle_t handle,
                                                     int m,
                                                     int n,
                                                     int* len) {
    return cusolverDnSgesvd_bufferSize(handle, m, n, len);
}

template <>
inline cusolverStatus_t gesvd_cuda_buffersize<double>(cusolverDnHandle_t handle,
                                                      int m,
                                                      int n,
                                                      int* len) {
    return cusolverDnDgesvd_bufferSize(handle, m, n, len);
}

template <>
inline cusolverStatus_t gesvd_cuda<float>(cusolverDnHandle_t handle,
                                          char jobu,
                                          char jobvt,
                                          int m,
                                          int n,
                                          float* A,
                                          int lda,
                                          float* S,
                                          float* U,
                                          int ldu,
                                          float* VT,
                                          int ldvt,
                                          float* workspace,
                                          int len,
                                          float* rwork,
                                          int* dinfo) {
    return cusolverDnSgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT,
                            ldvt, workspace, len, rwork, dinfo);
}

template <>
inline cusolverStatus_t gesvd_cuda<double>(cusolverDnHandle_t handle,
                                           char jobu,
                                           char jobvt,
                                           int m,
                                           int n,
                                           double* A,
                                           int lda,
                                           double* S,
                                           double* U,
                                           int ldu,
                                           double* VT,
                                           int ldvt,
                                           double* workspace,
                                           int len,
                                           double* rwork,
                                           int* dinfo) {
    return cusolverDnDgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT,
                            ldvt, workspace, len, rwork, dinfo);
}

#endif
}  // namespace core
}  // namespace open3d
