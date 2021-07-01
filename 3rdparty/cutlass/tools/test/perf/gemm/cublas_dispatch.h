/***************************************************************************************************
 * Copyright (c) 2017-2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cutlass/matrix_traits.h"
#include "tools/util/type_traits.h"

namespace perf {

/// Dispatcher for cuBLAS kernels
template <typename AType, typename BType, typename CType, typename Accumulator, typename Scalar>
struct CublasGemmDispatch {
  /// Type used for device-side allocations
  typedef typename cutlass::TypeTraits<AType>::device_type ADeviceType;
  typedef typename cutlass::TypeTraits<BType>::device_type BDeviceType;
  typedef typename cutlass::TypeTraits<CType>::device_type CDeviceType;
  typedef typename cutlass::TypeTraits<Accumulator>::device_type AccumulatorDeviceType;
  typedef typename cutlass::TypeTraits<Scalar>::device_type ScalarDeviceType;

  static cublasOperation_t convert(cutlass::MatrixLayout::Kind layout) {
    switch (layout) {
      case cutlass::MatrixLayout::kRowMajor:
        return CUBLAS_OP_T;
      case cutlass::MatrixLayout::kColumnMajor:
        return CUBLAS_OP_N;
      default:
        break;
    }
    return CUBLAS_OP_N;
  }

  /// Launches a cuBLAS GEMM kernel
  cublasStatus_t operator()(cublasHandle_t handle,
                            cutlass::MatrixLayout::Kind layout_a,
                            cutlass::MatrixLayout::Kind layout_b,
                            int m,
                            int n,
                            int k,
                            Scalar alpha,
                            const ADeviceType *A,
                            int lda,
                            const BDeviceType *B,
                            int ldb,
                            Scalar beta,
                            CDeviceType *C,
                            int ldc,
                            cublasGemmAlgo_t algorithm) {
    #if CUTLASS_ENABLE_CUBLAS
    return cublasGemmEx(handle,
                        convert(layout_a),
                        convert(layout_b),
                        m,
                        n,
                        k,
                        reinterpret_cast<ScalarDeviceType const *>(&alpha),
                        A,
                        cutlass::TypeTraits<ADeviceType>::cublas_type,
                        lda,
                        B,
                        cutlass::TypeTraits<BDeviceType>::cublas_type,
                        ldb,
                        reinterpret_cast<ScalarDeviceType const *>(&beta),
                        C,
                        cutlass::TypeTraits<CDeviceType>::cublas_type,
                        ldc,
                        cutlass::TypeTraits<AccumulatorDeviceType>::cublas_type,
                        algorithm);
    #else
    return CUBLAS_STATUS_NOT_SUPPORTED;
    #endif
  }
};

/// Dispatcher for batched strided cuBLAS kernels
template <typename AType, typename BType, typename CType, typename Accumulator, typename Scalar>
struct CublasBatchedStridedGemmDispatch {
  /// Type used for device-side allocations
  typedef typename cutlass::TypeTraits<AType>::device_type ADeviceType;
  typedef typename cutlass::TypeTraits<BType>::device_type BDeviceType;
  typedef typename cutlass::TypeTraits<CType>::device_type CDeviceType;
  typedef typename cutlass::TypeTraits<Accumulator>::device_type AccumulatorDeviceType;
  typedef typename cutlass::TypeTraits<Scalar>::device_type ScalarDeviceType;

  static cublasOperation_t convert(cutlass::MatrixLayout::Kind layout) {
    switch (layout) {
    case cutlass::MatrixLayout::kRowMajor:
      return CUBLAS_OP_T;
    case cutlass::MatrixLayout::kColumnMajor:
      return CUBLAS_OP_N;
    default:
      break;
    }
    return CUBLAS_OP_N;
  }

  /// Launches a cuBLAS GEMM kernel
  cublasStatus_t operator()(cublasHandle_t handle,
    cutlass::MatrixLayout::Kind layout_a,
    cutlass::MatrixLayout::Kind layout_b,
    int m,
    int n,
    int k,
    Scalar alpha,
    const ADeviceType *A,
    int lda,
    long long int batch_stride_A,
    const BDeviceType *B,
    int ldb,
    long long int batch_stride_B,
    Scalar beta,
    CDeviceType *C,
    int ldc,
    long long int batch_stride_C,
    int batch_count,
    cublasGemmAlgo_t algorithm) {
    #if CUTLASS_ENABLE_CUBLAS && defined(CUDA_VERSION) && CUDA_VERSION >= 9010
    return cublasGemmStridedBatchedEx(handle,
      convert(layout_a),
      convert(layout_b),
      m,
      n,
      k,
      reinterpret_cast<ScalarDeviceType const *>(&alpha),
      A,
      cutlass::TypeTraits<ADeviceType>::cublas_type,
      lda,
      batch_stride_A,
      B,
      cutlass::TypeTraits<BDeviceType>::cublas_type,
      ldb,
      batch_stride_B,
      reinterpret_cast<ScalarDeviceType const *>(&beta),
      C,
      cutlass::TypeTraits<CDeviceType>::cublas_type,
      ldc,
      batch_stride_C,
      batch_count,
      cutlass::TypeTraits<AccumulatorDeviceType>::cublas_type,
      algorithm);
    #else
    return CUBLAS_STATUS_NOT_SUPPORTED;
    #endif
  }
};

}  // namespace perf
