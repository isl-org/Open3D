// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// This file contains headers for BLAS/LAPACK implementations for CUDA.
//
// For developers, please make sure that this file is not ultimately included in
// Open3D.h.

#pragma once

#ifdef BUILD_CUDA_MODULE

#if defined(USE_HIP)
// hipBLAS / hipSOLVER expose a near-1:1 typed-Dense API for the cuBLAS /
// cuSOLVER calls the linalg sources use; alias the CUDA spellings to the HIP
// ones so the host .cpp wrappers compile unchanged. The cuSOLVER status enum
// is wider than hipSOLVER's, so the orphan cases below are guarded out.
#include <hipblas/hipblas.h>
#include <hipsolver/hipsolver.h>

#define cublasStatus_t hipblasStatus_t
#define cublasHandle_t hipblasHandle_t
#define cublasOperation_t hipblasOperation_t
#define cublasFillMode_t hipblasFillMode_t
#define cublasDiagType_t hipblasDiagType_t
#define cublasSideMode_t hipblasSideMode_t
#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasSgemm hipblasSgemm
#define cublasDgemm hipblasDgemm
#define cublasStrsm hipblasStrsm
#define cublasDtrsm hipblasDtrsm
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_SIDE_LEFT HIPBLAS_SIDE_LEFT
#define CUBLAS_FILL_MODE_UPPER HIPBLAS_FILL_MODE_UPPER
#define CUBLAS_DIAG_NON_UNIT HIPBLAS_DIAG_NON_UNIT
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_STATUS_NOT_SUPPORTED HIPBLAS_STATUS_NOT_SUPPORTED

#define cusolverStatus_t hipsolverStatus_t
#define cusolverDnHandle_t hipsolverDnHandle_t
#define cusolverDnCreate hipsolverDnCreate
#define cusolverDnDestroy hipsolverDnDestroy
#define cusolverDnSgetrf hipsolverDnSgetrf
#define cusolverDnDgetrf hipsolverDnDgetrf
#define cusolverDnSgetrf_bufferSize hipsolverDnSgetrf_bufferSize
#define cusolverDnDgetrf_bufferSize hipsolverDnDgetrf_bufferSize
#define cusolverDnSgetrs hipsolverDnSgetrs
#define cusolverDnDgetrs hipsolverDnDgetrs
#define cusolverDnSgesvd hipsolverDnSgesvd
#define cusolverDnDgesvd hipsolverDnDgesvd
#define cusolverDnSgesvd_bufferSize hipsolverDnSgesvd_bufferSize
#define cusolverDnDgesvd_bufferSize hipsolverDnDgesvd_bufferSize
#define cusolverDnSgeqrf hipsolverDnSgeqrf
#define cusolverDnDgeqrf hipsolverDnDgeqrf
#define cusolverDnSgeqrf_bufferSize hipsolverDnSgeqrf_bufferSize
#define cusolverDnDgeqrf_bufferSize hipsolverDnDgeqrf_bufferSize
#define cusolverDnSormqr hipsolverDnSormqr
#define cusolverDnDormqr hipsolverDnDormqr
#define cusolverDnSormqr_bufferSize hipsolverDnSormqr_bufferSize
#define cusolverDnDormqr_bufferSize hipsolverDnDormqr_bufferSize

#define CUSOLVER_STATUS_SUCCESS HIPSOLVER_STATUS_SUCCESS
#define CUSOLVER_STATUS_NOT_INITIALIZED HIPSOLVER_STATUS_NOT_INITIALIZED
#define CUSOLVER_STATUS_ALLOC_FAILED HIPSOLVER_STATUS_ALLOC_FAILED
#define CUSOLVER_STATUS_INVALID_VALUE HIPSOLVER_STATUS_INVALID_VALUE
#define CUSOLVER_STATUS_ARCH_MISMATCH HIPSOLVER_STATUS_ARCH_MISMATCH
#define CUSOLVER_STATUS_MAPPING_ERROR HIPSOLVER_STATUS_MAPPING_ERROR
#define CUSOLVER_STATUS_EXECUTION_FAILED HIPSOLVER_STATUS_EXECUTION_FAILED
#define CUSOLVER_STATUS_INTERNAL_ERROR HIPSOLVER_STATUS_INTERNAL_ERROR
#define CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED \
    HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED
#define CUSOLVER_STATUS_NOT_SUPPORTED HIPSOLVER_STATUS_NOT_SUPPORTED
#define CUSOLVER_STATUS_ZERO_PIVOT HIPSOLVER_STATUS_ZERO_PIVOT
#else
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusolver_common.h>
#endif

#include <fmt/core.h>
#include <fmt/format.h>

namespace fmt {

template <>
struct formatter<cusolverStatus_t> {
    template <typename FormatContext>
    auto format(const cusolverStatus_t& c,
                FormatContext& ctx) const -> decltype(ctx.out()) {
        const char* text = nullptr;
        switch (c) {
            case CUSOLVER_STATUS_SUCCESS:
                text = "CUSOLVER_STATUS_SUCCESS";
                break;
            case CUSOLVER_STATUS_NOT_INITIALIZED:
                text = "CUSOLVER_STATUS_NOT_INITIALIZED";
                break;
            case CUSOLVER_STATUS_ALLOC_FAILED:
                text = "CUSOLVER_STATUS_ALLOC_FAILED";
                break;
            case CUSOLVER_STATUS_INVALID_VALUE:
                text = "CUSOLVER_STATUS_INVALID_VALUE";
                break;
            case CUSOLVER_STATUS_ARCH_MISMATCH:
                text = "CUSOLVER_STATUS_ARCH_MISMATCH";
                break;
            case CUSOLVER_STATUS_MAPPING_ERROR:
                text = "CUSOLVER_STATUS_MAPPING_ERROR";
                break;
            case CUSOLVER_STATUS_EXECUTION_FAILED:
                text = "CUSOLVER_STATUS_EXECUTION_FAILED";
                break;
            case CUSOLVER_STATUS_INTERNAL_ERROR:
                text = "CUSOLVER_STATUS_INTERNAL_ERROR";
                break;
            case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
                text = "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
                break;
            case CUSOLVER_STATUS_NOT_SUPPORTED:
                text = "CUSOLVER_STATUS_NOT_SUPPORTED";
                break;
            case CUSOLVER_STATUS_ZERO_PIVOT:
                text = "CUSOLVER_STATUS_ZERO_PIVOT";
                break;
#if !defined(USE_HIP)
            // hipSOLVER's status enum does not define these codes.
            case CUSOLVER_STATUS_INVALID_LICENSE:
                text = "CUSOLVER_STATUS_INVALID_LICENSE";
                break;
            case CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED:
                text = "CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED";
                break;
            case CUSOLVER_STATUS_IRS_PARAMS_INVALID:
                text = "CUSOLVER_STATUS_IRS_PARAMS_INVALID";
                break;
            case CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC:
                text = "CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC";
                break;
            case CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE:
                text = "CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE";
                break;
            case CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER:
                text = "CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER";
                break;
            case CUSOLVER_STATUS_IRS_INTERNAL_ERROR:
                text = "CUSOLVER_STATUS_IRS_INTERNAL_ERROR";
                break;
            case CUSOLVER_STATUS_IRS_NOT_SUPPORTED:
                text = "CUSOLVER_STATUS_IRS_NOT_SUPPORTED";
                break;
            case CUSOLVER_STATUS_IRS_OUT_OF_RANGE:
                text = "CUSOLVER_STATUS_IRS_OUT_OF_RANGE";
                break;
            case CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES:
                text = "CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_"
                       "GMRES";
                break;
            case CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED:
                text = "CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED";
                break;
            case CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED:
                text = "CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED";
                break;
            case CUSOLVER_STATUS_IRS_MATRIX_SINGULAR:
                text = "CUSOLVER_STATUS_IRS_MATRIX_SINGULAR";
                break;
            case CUSOLVER_STATUS_INVALID_WORKSPACE:
                text = "CUSOLVER_STATUS_INVALID_WORKSPACE";
                break;
#endif  // !USE_HIP
            default:
                text = "CUSOLVER_STATUS_UNKNOWN";
                break;
        }
        return format_to(ctx.out(), text);
    }

    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx) -> decltype(ctx.begin()) {
        return ctx.begin();
    }
};

}  // namespace fmt

#endif
