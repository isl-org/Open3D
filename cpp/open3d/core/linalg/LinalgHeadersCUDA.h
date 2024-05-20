// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// This file contains headers for BLAS/LAPACK implementations for CUDA.
//
// For developers, please make sure that this file is not ultimately included in
// Open3D.h.

#pragma once

#ifdef BUILD_CUDA_MODULE
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusolver_common.h>
#include <fmt/core.h>
#include <fmt/format.h>

namespace fmt {

template <>
struct formatter<cusolverStatus_t> {
    template <typename FormatContext>
    auto format(const cusolverStatus_t& c, FormatContext& ctx) const
            -> decltype(ctx.out()) {
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
