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

#include "cutlass/wmma_matrix.h"
#ifdef CUTLASS_USE_WMMA_API

#pragma warning( disable : 4503)

////////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/wmma_gemm_traits.h"
#include "tools/test/perf/cutlass_perf_test.h"
#include "tools/test/perf/gemm/gemm_profiler.h"
#include "tools/test/perf/gemm/cutlass_dispatch.h"
#include "tools/test/perf/gemm/gemm_perf_testbed.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits>
struct WmmaGemmDispatch {

  typedef cutlass::gemm::Gemm<Traits> Gemm;

  typedef typename Gemm::Params Params;

  /// Indicate warp-level GEMM
  static bool const kThreadMultiplyAdd = false;

  #if CUTLASS_ENABLE_CUBLAS
  static bool const kRunCuBLAS = true;
  #else
  static bool const kRunCuBLAS = false;
  #endif

  static cutlass::MatrixLayout::Kind const kLayoutA = Traits::kLayoutA;
  static cutlass::MatrixLayout::Kind const kLayoutB = Traits::kLayoutB;

  typedef typename Traits::ScalarA ScalarA;
  typedef typename Traits::ScalarB ScalarB;
  typedef typename Traits::ScalarC ScalarC;
  typedef typename Traits::ScalarD ScalarD;
  typedef typename Traits::Epilogue::Functor::Scalar Scalar;

  //
  // Data members
  //

  /// Params argument
  Params params;

  //
  // Methods
  //

  WmmaGemmDispatch() {}

  /// Initializes params object
  WmmaGemmDispatch(
    int m,
    int n,
    int k,
    Scalar alpha,
    ScalarA const* d_a,
    int lda,
    ScalarB const* d_b,
    int ldb,
    Scalar beta,
    ScalarC const* d_c,
    int ldc,
    ScalarD* d_d,
    int ldd) {

    params.initialize(m, n, k, alpha, d_a, lda, d_b, ldb, beta, d_c, ldc, d_d, ldd);
  }

  WmmaGemmDispatch(int m,
    int n,
    int k,
    Scalar alpha,
    ScalarA const* d_a,
    int lda,
    long long int batch_stride_A,
    ScalarB const* d_b,
    int ldb,
    long long int batch_stride_B,
    Scalar beta,
    ScalarC const* d_c,
    int ldc,
    long long int batch_stride_C,
    ScalarD* d_d,
    int ldd,
    long long int batch_stride_D,
    int batch_count) {
    params.initialize(m, n, k, alpha, d_a, lda, batch_stride_A,
      d_b, ldb, batch_stride_B,
      beta, d_c, ldc, batch_stride_C,
      d_d, ldd, batch_stride_D,
      batch_count);
  }

  /// Initializes params object
  WmmaGemmDispatch(Params const& _params) : params(_params) {}

  /// Launches kernel
  cudaError_t operator()() { return Gemm::launch(params); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace perf {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename DummyT>
int profile_wmma_gemm_f32(TestbenchOutput<GemmProblem> &output, TestbenchOptions const &options, Config const &config) {
  typedef perf::GemmProfiler<cutlass::half_t, cutlass::half_t, float, float, float> GemmProfiler;

  int results = 0;

  {
    typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kRowMajor>
    WmmaGemmTraits;

    typedef WmmaGemmDispatch<WmmaGemmTraits> Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, "wmma_gemm_nt", options, config);
  }

  {
    typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kColumnMajor>
    WmmaGemmTraits;

    typedef WmmaGemmDispatch<WmmaGemmTraits> Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, "wmma_gemm_nn", options, config);
  }

  {
    typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kColumnMajor>
      WmmaGemmTraits;

    typedef WmmaGemmDispatch<WmmaGemmTraits> Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, "wmma_gemm_tn", options, config);
  }

  {
    typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kRowMajor>
      WmmaGemmTraits;

    typedef WmmaGemmDispatch<WmmaGemmTraits> Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, "wmma_gemm_tt", options, config);
  }

  return results;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename DummyT>
int profile_wmma_gemm_f16(
    TestbenchOutput<GemmProblem> &output,
    TestbenchOptions const &options,
    Config const &config) {

  typedef perf::GemmProfiler<
    cutlass::half_t,
    cutlass::half_t,
    cutlass::half_t,
    cutlass::half_t,
    cutlass::half_t> GemmProfiler;

  int results = 0;

  {
    typedef cutlass::gemm::WmmaGemmTraits<
      cutlass::MatrixLayout::kColumnMajor,
      cutlass::MatrixLayout::kRowMajor,
      cutlass::Shape<32, 256, 128>,
      half,
      half,
      half,
      cutlass::gemm::LinearScaling<half>,
      half,
      cutlass::Shape<32, 64, 64>
    >
      WmmaGemmTraits;

    typedef WmmaGemmDispatch<WmmaGemmTraits> Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, "wmma_gemm_f16_nt", options, config);
  }

  {
    typedef cutlass::gemm::WmmaGemmTraits<
      cutlass::MatrixLayout::kColumnMajor,
      cutlass::MatrixLayout::kColumnMajor,
      cutlass::Shape<32, 256, 128>,
      half,
      half,
      half,
      cutlass::gemm::LinearScaling<half>,
      half,
      cutlass::Shape<32, 64, 64>
    >
      WmmaGemmTraits;

    typedef WmmaGemmDispatch<WmmaGemmTraits> Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, "wmma_gemm_f16_nn", options, config);
  }

  {
    typedef cutlass::gemm::WmmaGemmTraits<
      cutlass::MatrixLayout::kRowMajor,
      cutlass::MatrixLayout::kColumnMajor,
      cutlass::Shape<32, 256, 128>,
      half,
      half,
      half,
      cutlass::gemm::LinearScaling<half>,
      half,
      cutlass::Shape<32, 64, 64>
    >
      WmmaGemmTraits;

    typedef WmmaGemmDispatch<WmmaGemmTraits> Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, "wmma_gemm_f16_tn", options, config);
  }

  {
    typedef cutlass::gemm::WmmaGemmTraits<
      cutlass::MatrixLayout::kRowMajor,
      cutlass::MatrixLayout::kRowMajor,
      cutlass::Shape<32, 256, 128>,
      half,
      half,
      half,
      cutlass::gemm::LinearScaling<half>,
      half,
      cutlass::Shape<32, 64, 64>
    >
      WmmaGemmTraits;

    typedef WmmaGemmDispatch<WmmaGemmTraits> Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, "wmma_gemm_f16_tt", options, config);
  }

  return results;
}


template <typename DummyT>
int profile_wmma_4_gemm_f16(
    TestbenchOutput<GemmProblem> &output,
    TestbenchOptions const &options,
    Config const &config) {

  typedef perf::GemmProfiler<
    cutlass::half_t,
    cutlass::half_t,
    cutlass::half_t,
    cutlass::half_t,
    cutlass::half_t> GemmProfiler;

  int results = 0;

  // a set of test requires leading dim to be multiple of 4 instead of 8

  {
    typedef cutlass::gemm::WmmaGemmTraits<
      cutlass::MatrixLayout::kColumnMajor,
      cutlass::MatrixLayout::kRowMajor,
      cutlass::Shape<32, 16, 16>,
      half,
      half,
      half,
      cutlass::gemm::LinearScaling<half>,
      half,
      typename cutlass::gemm::WmmaGemmAccumulatorsPerWarp<typename cutlass::Shape<32, 16, 16> >::Shape,
      typename cutlass::Shape<16, 16, 16>,
      4, /*kScalarsPerLdgA_*/
      4, /*kScalarsPerLdgB_*/
      4, /*KScalarsPerLdsA_*/
      4, /*KScalarsPerLdsB_*/
      4 / sizeof(half), /*kScalarsPerLdgCAndStgD_*/
      4 / sizeof(half), /*kScalarsPerStsD_*/
      4 / sizeof(half)  /*kScalarsPerLdsD_*/
    >
      WmmaGemmTraits;

    typedef WmmaGemmDispatch<WmmaGemmTraits> Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, "wmma_4_gemm_f16_nt", options, config);
  }

  {
    typedef cutlass::gemm::WmmaGemmTraits<
      cutlass::MatrixLayout::kColumnMajor,
      cutlass::MatrixLayout::kColumnMajor,
      cutlass::Shape<32, 16, 16>,
      half,
      half,
      half,
      cutlass::gemm::LinearScaling<half>,
      half,
      typename cutlass::gemm::WmmaGemmAccumulatorsPerWarp<typename cutlass::Shape<32, 16, 16> >::Shape,
      typename cutlass::Shape<16, 16, 16>,
      4, /*kScalarsPerLdgA_*/
      4, /*kScalarsPerLdgB_*/
      4, /*KScalarsPerLdsA_*/
      4, /*KScalarsPerLdsB_*/
      4 / sizeof(half), /*kScalarsPerLdgCAndStgD_*/
      4 / sizeof(half), /*kScalarsPerStsD_*/
      4 / sizeof(half)  /*kScalarsPerLdsD_*/
    >
      WmmaGemmTraits;

    typedef WmmaGemmDispatch<WmmaGemmTraits> Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, "wmma_4_gemm_f16_nn", options, config);
  }

  {
    typedef cutlass::gemm::WmmaGemmTraits<
      cutlass::MatrixLayout::kRowMajor,
      cutlass::MatrixLayout::kColumnMajor,
      cutlass::Shape<32, 16, 16>,
      half,
      half,
      half,
      cutlass::gemm::LinearScaling<half>,
      half,
      typename cutlass::gemm::WmmaGemmAccumulatorsPerWarp<typename cutlass::Shape<32, 16, 16> >::Shape,
      typename cutlass::Shape<16, 16, 16>,
      4, /*kScalarsPerLdgA_*/
      4, /*kScalarsPerLdgB_*/
      4, /*KScalarsPerLdsA_*/
      4, /*KScalarsPerLdsB_*/
      4 / sizeof(half), /*kScalarsPerLdgCAndStgD_*/
      4 / sizeof(half), /*kScalarsPerStsD_*/
      4 / sizeof(half)  /*kScalarsPerLdsD_*/
    >
      WmmaGemmTraits;

    typedef WmmaGemmDispatch<WmmaGemmTraits> Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, "wmma_4_gemm_f16_tn", options, config);
  }

  {
    typedef cutlass::gemm::WmmaGemmTraits<
      cutlass::MatrixLayout::kRowMajor,
      cutlass::MatrixLayout::kRowMajor,
      cutlass::Shape<32, 16, 16>,
      half,
      half,
      half,
      cutlass::gemm::LinearScaling<half>,
      half,
      typename cutlass::gemm::WmmaGemmAccumulatorsPerWarp<typename cutlass::Shape<32, 16, 16> >::Shape,
      typename cutlass::Shape<16, 16, 16>,
      4, /*kScalarsPerLdgA_*/
      4, /*kScalarsPerLdgB_*/
      4, /*KScalarsPerLdsA_*/
      4, /*KScalarsPerLdsB_*/
      4 / sizeof(half), /*kScalarsPerLdgCAndStgD_*/
      4 / sizeof(half), /*kScalarsPerStsD_*/
      4 / sizeof(half)  /*kScalarsPerLdsD_*/
    >
      WmmaGemmTraits;

    typedef WmmaGemmDispatch<WmmaGemmTraits> Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, "wmma_4_gemm_f16_tt", options, config);
  }

  return results;
}

template <typename DummyT>
int profile_wmma_4_fp16_sgemm_fp16(
  TestbenchOutput<GemmProblem> &output,
  TestbenchOptions const &options,
  Config const &config) {

  typedef perf::GemmProfiler<
    cutlass::half_t,
    cutlass::half_t,
    cutlass::half_t,
    float,
    float> GemmProfiler;

  int results = 0;

  // a set of test requires leading dim to be multiple of 4 instead of 8

  {
    typedef float accumu_type;
    typedef half c_type;
    typedef cutlass::gemm::WmmaGemmTraits<
      cutlass::MatrixLayout::kColumnMajor,
      cutlass::MatrixLayout::kRowMajor,
      cutlass::Shape<32, 16, 16>,
      half,
      half,
      c_type,
      cutlass::gemm::LinearScaling<accumu_type>,
      accumu_type,
      typename cutlass::gemm::WmmaGemmAccumulatorsPerWarp<typename cutlass::Shape<32, 16, 16> >::Shape,
      typename cutlass::Shape<16, 16, 16>,
      4, /*kScalarsPerLdgA_*/
      4, /*kScalarsPerLdgB_*/
      4, /*KScalarsPerLdsA_*/
      4, /*KScalarsPerLdsB_*/
      8 / sizeof(c_type), /*kScalarsPerLdgCAndStgD_*/
      8 / sizeof(accumu_type), /*kScalarsPerStsD_*/
      8 / sizeof(accumu_type)  /*kScalarsPerLdsD_*/
    >
      WmmaGemmTraits;

    typedef WmmaGemmDispatch<WmmaGemmTraits> Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, "wmma_4_fp16_sgemm_fp16_nt", options, config);
  }

  {
    typedef float accumu_type;
    typedef half c_type;
    typedef cutlass::gemm::WmmaGemmTraits<
      cutlass::MatrixLayout::kColumnMajor,
      cutlass::MatrixLayout::kColumnMajor,
      cutlass::Shape<32, 16, 16>,
      half,
      half,
      c_type,
      cutlass::gemm::LinearScaling<accumu_type>,
      accumu_type,
      typename cutlass::gemm::WmmaGemmAccumulatorsPerWarp<typename cutlass::Shape<32, 16, 16> >::Shape,
      typename cutlass::Shape<16, 16, 16>,
      4, /*kScalarsPerLdgA_*/
      4, /*kScalarsPerLdgB_*/
      4, /*KScalarsPerLdsA_*/
      4, /*KScalarsPerLdsB_*/
      8 / sizeof(c_type), /*kScalarsPerLdgCAndStgD_*/
      8 / sizeof(accumu_type), /*kScalarsPerStsD_*/
      8 / sizeof(accumu_type)  /*kScalarsPerLdsD_*/
    >
      WmmaGemmTraits;

    typedef WmmaGemmDispatch<WmmaGemmTraits> Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, "wmma_4_fp16_sgemm_fp16_nn", options, config);
  }

  {
    typedef float accumu_type;
    typedef half c_type;
    typedef cutlass::gemm::WmmaGemmTraits<
      cutlass::MatrixLayout::kRowMajor,
      cutlass::MatrixLayout::kColumnMajor,
      cutlass::Shape<32, 16, 16>,
      half,
      half,
      c_type,
      cutlass::gemm::LinearScaling<accumu_type>,
      accumu_type,
      typename cutlass::gemm::WmmaGemmAccumulatorsPerWarp<typename cutlass::Shape<32, 16, 16> >::Shape,
      typename cutlass::Shape<16, 16, 16>,
      4, /*kScalarsPerLdgA_*/
      4, /*kScalarsPerLdgB_*/
      4, /*KScalarsPerLdsA_*/
      4, /*KScalarsPerLdsB_*/
      8 / sizeof(c_type), /*kScalarsPerLdgCAndStgD_*/
      8 / sizeof(accumu_type), /*kScalarsPerStsD_*/
      8 / sizeof(accumu_type)  /*kScalarsPerLdsD_*/
    >
      WmmaGemmTraits;

    typedef WmmaGemmDispatch<WmmaGemmTraits> Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, "wmma_4_fp16_sgemm_fp16_tn", options, config);
  }

  {
    typedef float accumu_type;
    typedef half c_type;
    typedef cutlass::gemm::WmmaGemmTraits<
      cutlass::MatrixLayout::kRowMajor,
      cutlass::MatrixLayout::kRowMajor,
      cutlass::Shape<32, 16, 16>,
      half,
      half,
      c_type,
      cutlass::gemm::LinearScaling<accumu_type>,
      accumu_type,
      typename cutlass::gemm::WmmaGemmAccumulatorsPerWarp<typename cutlass::Shape<32, 16, 16> >::Shape,
      typename cutlass::Shape<16, 16, 16>,
      4, /*kScalarsPerLdgA_*/
      4, /*kScalarsPerLdgB_*/
      4, /*KScalarsPerLdsA_*/
      4, /*KScalarsPerLdsB_*/
      8 / sizeof(c_type), /*kScalarsPerLdgCAndStgD_*/
      8 / sizeof(accumu_type), /*kScalarsPerStsD_*/
      8 / sizeof(accumu_type)  /*kScalarsPerLdsD_*/
    >
      WmmaGemmTraits;

    typedef WmmaGemmDispatch<WmmaGemmTraits> Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, "wmma_4_fp16_sgemm_fp16_tt", options, config);
  }

  return results;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

struct WmmaGemmRegistrar {
  WmmaGemmRegistrar() {
    RegisterGemmProfileFunc(profile_wmma_gemm_f32<void>);
    RegisterGemmProfileFunc(profile_wmma_gemm_f16<void>);

//#ifdef EXHAUSTIVE_PROF
    RegisterGemmProfileFunc(profile_wmma_4_gemm_f16<void>);
    //fp32 accum with fp16 input and output
    RegisterGemmProfileFunc(profile_wmma_4_fp16_sgemm_fp16<void>);
//#endif  // defined EXHAUSTIVE_PROF
  }
};

volatile WmmaGemmRegistrar _WmmaGemmRegistrar;

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace perf

////////////////////////////////////////////////////////////////////////////////////////////////////

#endif  // defined CUTLASS_USE_WMMA_API
