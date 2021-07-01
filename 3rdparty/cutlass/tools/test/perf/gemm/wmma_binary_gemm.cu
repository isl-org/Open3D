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

#include "tools/test/perf/cutlass_perf_test.h"
#include "tools/test/perf/gemm/gemm_profiler.h"
#include "tools/test/perf/gemm/gemm_perf_testbed.h"

#include "cutlass/wmma_matrix.h"
#ifdef CUTLASS_USE_WMMA_API
#ifdef CUTLASS_USE_SUBBYTE_WMMA
#pragma warning( disable : 4503)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/wmma_gemm_traits.h"
#include "tools/test/perf/gemm/cutlass_dispatch.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Traits>
struct WmmaBinaryGemmDispatch {

  typedef cutlass::gemm::Gemm<Traits> Gemm;

  typedef typename Gemm::Params Params;

  /// Indicate warp-level GEMM
  static bool const kThreadMultiplyAdd = false;

  static bool const kRunCuBLAS = false;

  static cutlass::MatrixLayout::Kind const kLayoutA = Traits::kLayoutA;
  static cutlass::MatrixLayout::Kind const kLayoutB = Traits::kLayoutB;

  //
  // Data members
  //

  /// Params argument
  Params params;

  //
  // Methods
  //

  WmmaBinaryGemmDispatch() {}

  /// Initializes params object
  WmmaBinaryGemmDispatch(int m, int n, int k, int alpha,
                       cutlass::Vector<cutlass::bin1_t, 32> const* d_a, int lda,
                       cutlass::Vector<cutlass::bin1_t, 32> const* d_b, int ldb, int beta,
                       int const* d_c, int ldc, int* d_d, int ldd) {

    params.initialize(m, n, k * 32, alpha, d_a, lda, d_b, ldb, beta, d_c, ldc, d_d, ldd);
  }

  /// batched strided bmma
  WmmaBinaryGemmDispatch(int m, int n, int k, int alpha,
    cutlass::Vector<cutlass::bin1_t, 32> const* d_a, int lda, long long int batch_stride_a, 
    cutlass::Vector<cutlass::bin1_t, 32> const* d_b, int ldb, long long int batch_stride_b, int beta,
    int const* d_c, int ldc, long long int batch_stride_c, int* d_d, int ldd, long long int batch_stride_d,
    int batch_count) {
    assert(0);
  }

  /// Initializes params object
  WmmaBinaryGemmDispatch(Params const& _params) : params(_params) {}

  /// Launches kernel
  cudaError_t operator()() { return Gemm::launch(params); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace perf {

////////////////////////////////////////////////////////////////////////////////////////////////////

int profile_wmma_binary_gemm(TestbenchOutput<GemmProblem> &output, TestbenchOptions const &options, Config const &config) {
  typedef perf::GemmProfiler<cutlass::Vector<cutlass::bin1_t, 32>, cutlass::Vector<cutlass::bin1_t, 32>, int, int, int> GemmProfiler;

  int results = 0;

  // compute capability check
  if (!options.compute_capability_exact(7, 5)) {
    return 0;
  }

  {
    typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                          cutlass::MatrixLayout::kColumnMajor,
                                          cutlass::Shape<1024, 128, 128>,
                                          cutlass::Vector<cutlass::bin1_t, 32>,
                                          cutlass::Vector<cutlass::bin1_t, 32>,
                                          int,
                                          cutlass::gemm::LinearScaling<int>,
                                          int,
                                          cutlass::Shape<1024, 32, 64>,
                                          cutlass::Shape<128, 8, 8>,
                                          128,
                                          128>
      WmmaGemmTraits;

    typedef WmmaBinaryGemmDispatch<WmmaGemmTraits> Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, "wmma_binary_gemm_tn", options, config);
  }

  return results;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace perf

////////////////////////////////////////////////////////////////////////////////////////////////////

#else // ! CUTLASS_USE_SUBBYTE_WMMA

namespace perf {

int profile_wmma_binary_gemm(TestbenchOutput<GemmProblem> &output, TestbenchOptions const &options, Config const &config) {
  return 0;
}

} // namespace perf

#endif

struct WmmaBinaryGemmRegistrar {
  WmmaBinaryGemmRegistrar() { perf::RegisterGemmProfileFunc(perf::profile_wmma_binary_gemm); }
};

volatile WmmaBinaryGemmRegistrar _WmmaBinaryGemmRegistrar;

#endif // CUTLASS_USE_WMMA_API
