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

template <typename Gemm_,
          typename Index_,
          typename ScalarA_,
          typename ScalarB_,
          typename ScalarC_,
          typename ScalarD_,
          typename Compute_,
          typename ScalarEpilogue_,
          bool ThreadMultiplyAdd_,
          #if CUTLASS_ENABLE_CUBLAS
          bool RunCuBLAS_ = true
          #else
          bool RunCuBLAS_ = false
          #endif
>
struct CutlassDispatch {
  typedef typename Gemm_::Params Params;
  typedef Gemm_ Gemm;
  typedef Index_ Index;
  typedef ScalarA_ ScalarA;
  typedef ScalarB_ ScalarB;
  typedef ScalarC_ ScalarC;
  typedef ScalarD_ ScalarD;
  typedef Compute_ Compute;
  typedef ScalarEpilogue_ ScalarEpilogue;

  static bool const kThreadMultiplyAdd = ThreadMultiplyAdd_;
  static bool const kRunCuBLAS = RunCuBLAS_;

  static cutlass::MatrixLayout::Kind const kLayoutA = Gemm::Traits::kLayoutA;
  static cutlass::MatrixLayout::Kind const kLayoutB = Gemm::Traits::kLayoutB;

  //
  // Data members
  //

  /// Params argument
  Params params;

  //
  // Methods
  //

  // CutlassDispatch() {}

  /// Initializes params object
  CutlassDispatch(Index m,
                  Index n,
                  Index k,
                  ScalarEpilogue alpha,
                  ScalarA const* d_a,
                  Index lda,
                  ScalarB const* d_b,
                  Index ldb,
                  ScalarEpilogue beta,
                  ScalarC const* d_c,
                  Index ldc,
                  ScalarD* d_d,
                  Index ldd) {
    params.initialize(m, n, k, alpha, d_a, lda, d_b, ldb, beta, d_c, ldc, d_d, ldd);
  }

  /// Initializes batched strided params object
  CutlassDispatch(Index m,
                  Index n,
                  Index k,
                  ScalarEpilogue alpha,
                  ScalarA const* d_a,
                  Index lda, 
                  long long int batch_stride_A,
                  ScalarB const* d_b,
                  Index ldb,
                  long long int batch_stride_B,
                  ScalarEpilogue beta,
                  ScalarC const* d_c,
                  Index ldc,
                  long long int batch_stride_C,
                  ScalarD* d_d,
                  Index ldd,
                  long long int batch_stride_D,
                  Index batch_count) {
    params.initialize(m, n, k, alpha, d_a, lda, batch_stride_A, 
      d_b, ldb, batch_stride_B, 
      beta, d_c, ldc, batch_stride_C,
      d_d, ldd, batch_stride_D,
      batch_count);
  }

  /// Initializes params object
  CutlassDispatch(Params const& _params) : params(_params) {}

  /// Launches kernel
  cudaError_t operator()() { return Gemm::launch(params); }
};

/// Basic dispatcher inferred from GEMM traits
template <typename Traits>
struct CutlassDispatchBasic {
  /// Gemm kernel
  typedef cutlass::gemm::Gemm<Traits> Gemm;

  /// Index type
  typedef typename Traits::Index Index;

  /// The scalar for A.
  typedef typename Traits::ScalarA ScalarA;
  /// The scalar for B.
  typedef typename Traits::ScalarB ScalarB;
  /// The scalar for C.
  typedef typename Traits::ScalarC ScalarC;
  /// The scalar for D.
  typedef typename Traits::ScalarD ScalarD;
  typedef ScalarD Compute;
  typedef Compute ScalarEpilogue;

  typedef CutlassDispatch<Gemm,
                          Index,
                          ScalarA,
                          ScalarB,
                          ScalarC,
                          ScalarD,
                          Compute,
                          ScalarEpilogue,
                          true>
      Dispatch;
};
